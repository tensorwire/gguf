package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

// The streaming writer's output must be readable by the normal reader — that is
// the entire contract, and it is easy to break silently because a file with
// shifted offsets still parses.
func TestStreamingWriterRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")

	metas := []TensorMeta{
		{Name: "b.weight", Shape: []int{2, 3}, Elems: 6},
		{Name: "a.weight", Shape: []int{4}, Elems: 4},
	}
	w, err := NewStreamingSafeTensorsWriter(path, metas)
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	// Names reports sorted order, which is the order writes must follow.
	names := w.Names()
	if names[0] != "a.weight" || names[1] != "b.weight" {
		t.Fatalf("names not sorted: %v", names)
	}

	aData := []float32{1, 2, 3, 4}
	bData := []float32{10, 20, 30, 40, 50, 60}
	if err := w.WriteTensor(aData); err != nil {
		t.Fatalf("write a: %v", err)
	}
	if err := w.WriteTensor(bData); err != nil {
		t.Fatalf("write b: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	got, _, err := st.ReadTensorFloat32("a.weight")
	if err != nil {
		t.Fatalf("read a: %v", err)
	}
	for i := range aData {
		if got[i] != aData[i] {
			t.Fatalf("a[%d] = %v, want %v", i, got[i], aData[i])
		}
	}

	got, ti, err := st.ReadTensorFloat32("b.weight")
	if err != nil {
		t.Fatalf("read b: %v", err)
	}
	for i := range bData {
		if got[i] != bData[i] {
			t.Fatalf("b[%d] = %v, want %v", i, got[i], bData[i])
		}
	}
	if len(ti.Shape) != 2 || ti.Shape[0] != 2 || ti.Shape[1] != 3 {
		t.Errorf("b shape = %v, want [2 3]", ti.Shape)
	}
}

// A wrong element count would shift every later tensor relative to the header
// offsets already on disk. Readers cannot detect that — they return a
// neighbouring tensor's bytes — so it must fail at write time.
func TestStreamingWriterRejectsWrongElementCount(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.safetensors")
	w, err := NewStreamingSafeTensorsWriter(path, []TensorMeta{
		{Name: "x", Shape: []int{4}, Elems: 4},
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer w.Close()

	if err := w.WriteTensor([]float32{1, 2, 3}); err == nil {
		t.Error("short write accepted; offsets would be corrupt")
	}
}

// A truncated file still parses, so Close is the only place a missing tensor
// can be caught.
func TestStreamingWriterRejectsMissingTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.safetensors")
	w, err := NewStreamingSafeTensorsWriter(path, []TensorMeta{
		{Name: "x", Shape: []int{2}, Elems: 2},
		{Name: "y", Shape: []int{2}, Elems: 2},
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := w.WriteTensor([]float32{1, 2}); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := w.Close(); err == nil {
		t.Error("Close accepted a file missing its last tensor")
	}
}

// Shape and Elems disagreeing means the header describes a file the data does
// not fill.
func TestStreamingWriterRejectsShapeElemsMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.safetensors")
	_, err := NewStreamingSafeTensorsWriter(path, []TensorMeta{
		{Name: "x", Shape: []int{2, 3}, Elems: 5},
	})
	if err == nil {
		t.Error("accepted shape [2 3] with Elems=5")
	}
}

// Streaming exists so peak memory does not scale with model size; a tensor far
// larger than the internal chunk buffer must still write correctly.
func TestStreamingWriterHandlesTensorLargerThanChunk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.safetensors")
	const n = (1 << 16) + 1234 // deliberately not a chunk multiple
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%97) * 0.5
	}

	w, err := NewStreamingSafeTensorsWriter(path, []TensorMeta{
		{Name: "big", Shape: []int{n}, Elems: n},
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := w.WriteTensor(data); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	got, _, err := st.ReadTensorFloat32("big")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != n {
		t.Fatalf("read %d elements, want %d", len(got), n)
	}
	for _, i := range []int{0, 1, 65535, 65536, n - 1} {
		if got[i] != data[i] {
			t.Fatalf("element %d = %v, want %v", i, got[i], data[i])
		}
	}
	if fi, err := os.Stat(path); err == nil && fi.Size() < int64(n)*4 {
		t.Errorf("file is %d bytes, smaller than the tensor data alone", fi.Size())
	}
}
