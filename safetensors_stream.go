package gguf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// TensorMeta declares one tensor ahead of writing it.
//
// Elems is carried explicitly rather than derived from Shape so that a caller
// that already knows the element count does not have to recompute it, and so a
// mismatch between the two is caught at write time instead of corrupting the
// file.
type TensorMeta struct {
	Name  string
	Shape []int
	Elems int
}

// StreamingSafeTensorsWriter writes a .safetensors file one tensor at a time.
//
// SaveSafeTensors needs every tensor in memory at once, which is not viable for
// the merge and export paths: a 3B model in FP32 is ~13 GB, and merging holds
// the base weights too. This writer needs only the metadata up front — from
// which the header, and therefore every data offset, is fully determined — so
// tensors can be read, transformed, and released one at a time.
//
// The tradeoff is that the tensor set is fixed at construction: sizes are baked
// into the header before any data is written. WriteTensor must be called once
// per declared tensor, in the order the writer reports.
type StreamingSafeTensorsWriter struct {
	f       *os.File
	metas   []TensorMeta
	next    int    // index of the tensor expected next
	written uint64 // bytes of tensor data written so far
	dataOff uint64 // file offset where tensor data begins
}

// NewStreamingSafeTensorsWriter creates the file and writes its header.
//
// Tensors are sorted by name, matching SaveSafeTensors, so output is
// deterministic and independent of map iteration order. WriteTensor calls must
// follow that sorted order — see Names.
func NewStreamingSafeTensorsWriter(path string, metas []TensorMeta) (*StreamingSafeTensorsWriter, error) {
	if len(metas) == 0 {
		return nil, fmt.Errorf("safetensors: no tensors declared")
	}

	sorted := make([]TensorMeta, len(metas))
	copy(sorted, metas)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	type headerEntry struct {
		Dtype       string    `json:"dtype"`
		Shape       []int     `json:"shape"`
		DataOffsets [2]uint64 `json:"data_offsets"`
	}
	header := make(map[string]headerEntry, len(sorted))

	var offset uint64
	for i, m := range sorted {
		if m.Name == "" {
			return nil, fmt.Errorf("safetensors: tensor %d has no name", i)
		}
		if m.Elems < 0 {
			return nil, fmt.Errorf("safetensors: %s has negative element count %d", m.Name, m.Elems)
		}
		// Shape and Elems must agree, or the header will describe a file the
		// data does not fill and readers will silently return garbage.
		if len(m.Shape) > 0 {
			want := 1
			for _, d := range m.Shape {
				want *= d
			}
			if want != m.Elems {
				return nil, fmt.Errorf("safetensors: %s shape %v implies %d elements, got Elems=%d",
					m.Name, m.Shape, want, m.Elems)
			}
		}
		byteLen := uint64(m.Elems) * 4 // F32
		header[m.Name] = headerEntry{
			Dtype:       "F32",
			Shape:       m.Shape,
			DataOffsets: [2]uint64{offset, offset + byteLen},
		}
		offset += byteLen
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("safetensors: marshal header: %w", err)
	}

	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		f.Close()
		return nil, err
	}
	if _, err := f.Write(headerJSON); err != nil {
		f.Close()
		return nil, err
	}

	return &StreamingSafeTensorsWriter{
		f:       f,
		metas:   sorted,
		dataOff: uint64(8 + len(headerJSON)),
	}, nil
}

// Names returns the tensor names in the order WriteTensor expects them.
func (w *StreamingSafeTensorsWriter) Names() []string {
	out := make([]string, len(w.metas))
	for i, m := range w.metas {
		out[i] = m.Name
	}
	return out
}

// WriteTensor appends the next tensor's data.
//
// The element count must match what was declared: a short or long write would
// shift every subsequent tensor relative to the header offsets already on disk,
// which readers cannot detect — they would return data from a neighbouring
// tensor rather than fail.
func (w *StreamingSafeTensorsWriter) WriteTensor(data []float32) error {
	if w.f == nil {
		return fmt.Errorf("safetensors: writer is closed")
	}
	if w.next >= len(w.metas) {
		return fmt.Errorf("safetensors: too many tensors written (expected %d)", len(w.metas))
	}
	m := w.metas[w.next]
	if len(data) != m.Elems {
		return fmt.Errorf("safetensors: %s declared %d elements, got %d",
			m.Name, m.Elems, len(data))
	}

	// Convert in bounded chunks so peak memory stays flat regardless of tensor
	// size — a 100M-element tensor would otherwise need a 400 MB scratch buffer,
	// defeating the point of streaming.
	const chunkElems = 1 << 16
	buf := make([]byte, chunkElems*4)
	for off := 0; off < len(data); off += chunkElems {
		end := off + chunkElems
		if end > len(data) {
			end = len(data)
		}
		out := buf[:(end-off)*4]
		for i, v := range data[off:end] {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
		if _, err := w.f.Write(out); err != nil {
			return fmt.Errorf("safetensors: write %s: %w", m.Name, err)
		}
	}

	w.written += uint64(m.Elems) * 4
	w.next++
	return nil
}

// Close finishes the file, verifying that every declared tensor was written.
//
// A file missing its trailing tensors still parses — the header is valid and
// the offsets look plausible — so the check has to happen here or a truncated
// model ships silently.
func (w *StreamingSafeTensorsWriter) Close() error {
	if w.f == nil {
		return nil
	}
	f := w.f
	w.f = nil

	if w.next != len(w.metas) {
		f.Close()
		return fmt.Errorf("safetensors: only %d of %d tensors written",
			w.next, len(w.metas))
	}
	return f.Close()
}
