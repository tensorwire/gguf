package gguf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// SafeTensors reads tensor data from HuggingFace .safetensors files.
// Supports single files and sharded models. Streaming — only reads the
// tensors you ask for, doesn't load the whole file into memory.
//
// Zero Python. This is the bridge from HuggingFace to Go.
type SafeTensors struct {
	// Single file or shard directory
	dir    string
	single string // non-empty if single file

	// Parsed headers per shard file
	shards map[string]*shardInfo

	// Weight map: tensor name → shard filename
	weightMap map[string]string

	// All tensor names in order
	TensorNames []string
}

type shardInfo struct {
	path       string
	headerSize uint64
	tensors    map[string]*TensorInfo
}

// TensorInfo describes a tensor in a safetensors file.
type TensorInfo struct {
	Name        string
	Dtype       string    // "F32", "F16", "BF16", "I32", etc.
	Shape       []int
	DataOffsets [2]uint64 // [begin, end] relative to data buffer start
	ShardFile   string    // which shard file contains this tensor
}

// ByteSize returns the number of bytes for this tensor's data.
func (ti *TensorInfo) ByteSize() uint64 {
	return ti.DataOffsets[1] - ti.DataOffsets[0]
}

// NumElements returns the total number of elements.
func (ti *TensorInfo) NumElements() int {
	n := 1
	for _, s := range ti.Shape {
		n *= s
	}
	return n
}

// DtypeBytes returns bytes per element for a dtype string.
func DtypeBytes(dtype string) int {
	switch dtype {
	case "F64", "I64", "U64", "C64":
		return 8
	case "F32", "I32", "U32":
		return 4
	case "F16", "BF16", "I16", "U16":
		return 2
	case "I8", "U8", "BOOL", "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ", "F8_E5M2FNUZ":
		return 1
	default:
		return 0
	}
}

// OpenSafeTensors opens a safetensors file or sharded model directory.
// Pass a .safetensors file path or a directory containing shards + index.json.
func OpenSafeTensors(path string) (*SafeTensors, error) {
	st := &SafeTensors{
		shards:    make(map[string]*shardInfo),
		weightMap: make(map[string]string),
	}

	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		// Single file
		st.single = path
		st.dir = filepath.Dir(path)
		si, err := parseShardHeader(path)
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
		st.shards[filepath.Base(path)] = si
		for name := range si.tensors {
			st.weightMap[name] = filepath.Base(path)
			st.TensorNames = append(st.TensorNames, name)
		}
	} else {
		// Directory — look for index or single file
		st.dir = path
		indexPath := filepath.Join(path, "model.safetensors.index.json")
		singlePath := filepath.Join(path, "model.safetensors")

		if _, err := os.Stat(indexPath); err == nil {
			// Sharded model
			if err := st.loadIndex(indexPath); err != nil {
				return nil, err
			}
		} else if _, err := os.Stat(singlePath); err == nil {
			// Single model.safetensors
			st.single = singlePath
			si, err := parseShardHeader(singlePath)
			if err != nil {
				return nil, fmt.Errorf("parse %s: %w", singlePath, err)
			}
			st.shards[filepath.Base(singlePath)] = si
			for name := range si.tensors {
				st.weightMap[name] = "model.safetensors"
				st.TensorNames = append(st.TensorNames, name)
			}
		} else {
			return nil, fmt.Errorf("no .safetensors or index.json found in %s", path)
		}
	}

	sort.Strings(st.TensorNames)
	return st, nil
}

func (st *SafeTensors) loadIndex(indexPath string) error {
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return err
	}

	var idx struct {
		Metadata  map[string]interface{} `json:"metadata"`
		WeightMap map[string]string      `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &idx); err != nil {
		return fmt.Errorf("parse index: %w", err)
	}

	// Build weight map and collect shard filenames
	shardFiles := make(map[string]bool)
	for tensorName, shardFile := range idx.WeightMap {
		st.weightMap[tensorName] = shardFile
		st.TensorNames = append(st.TensorNames, tensorName)
		shardFiles[shardFile] = true
	}

	// Parse each shard header (only the header — don't read tensor data)
	for shardFile := range shardFiles {
		shardPath := filepath.Join(st.dir, shardFile)
		si, err := parseShardHeader(shardPath)
		if err != nil {
			return fmt.Errorf("parse shard %s: %w", shardFile, err)
		}
		st.shards[shardFile] = si
	}

	return nil
}

// parseShardHeader reads only the 8-byte length + JSON header from a safetensors file.
func parseShardHeader(path string) (*shardInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read header size (8 bytes, little-endian uint64)
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("read header size: %w", err)
	}
	if headerSize > 100_000_000 {
		return nil, fmt.Errorf("header too large: %d bytes", headerSize)
	}

	// Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Parse JSON — raw message to handle __metadata__ separately
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, fmt.Errorf("parse header JSON: %w", err)
	}

	si := &shardInfo{
		path:       path,
		headerSize: headerSize,
		tensors:    make(map[string]*TensorInfo),
	}

	for key, val := range raw {
		if key == "__metadata__" {
			continue
		}

		var ti struct {
			Dtype       string    `json:"dtype"`
			Shape       []int     `json:"shape"`
			DataOffsets [2]uint64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(val, &ti); err != nil {
			return nil, fmt.Errorf("parse tensor %s: %w", key, err)
		}

		si.tensors[key] = &TensorInfo{
			Name:        key,
			Dtype:       ti.Dtype,
			Shape:       ti.Shape,
			DataOffsets: ti.DataOffsets,
			ShardFile:   filepath.Base(path),
		}
	}

	return si, nil
}

// GetInfo returns metadata for a tensor by name.
func (st *SafeTensors) GetInfo(name string) (*TensorInfo, error) {
	shardFile, ok := st.weightMap[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found", name)
	}
	si, ok := st.shards[shardFile]
	if !ok {
		return nil, fmt.Errorf("shard %q not loaded", shardFile)
	}
	ti, ok := si.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not in shard %q", name, shardFile)
	}
	return ti, nil
}

// ReadTensorFloat32 reads a tensor and returns it as []float32.
// Handles F32 (native), F16 (conversion), BF16 (conversion).
func (st *SafeTensors) ReadTensorFloat32(name string) ([]float32, *TensorInfo, error) {
	ti, err := st.GetInfo(name)
	if err != nil {
		return nil, nil, err
	}

	shardFile := st.weightMap[name]
	si := st.shards[shardFile]

	// Open file and seek to tensor data
	f, err := os.Open(si.path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	dataStart := int64(8 + si.headerSize + ti.DataOffsets[0])
	dataLen := ti.DataOffsets[1] - ti.DataOffsets[0]

	if _, err := f.Seek(dataStart, io.SeekStart); err != nil {
		return nil, nil, fmt.Errorf("seek: %w", err)
	}

	rawBytes := make([]byte, dataLen)
	if _, err := io.ReadFull(f, rawBytes); err != nil {
		return nil, nil, fmt.Errorf("read tensor data: %w", err)
	}

	nelem := ti.NumElements()
	out := make([]float32, nelem)

	switch ti.Dtype {
	case "F32":
		for i := 0; i < nelem; i++ {
			bits := binary.LittleEndian.Uint32(rawBytes[i*4:])
			out[i] = math.Float32frombits(bits)
		}
	case "F16":
		for i := 0; i < nelem; i++ {
			bits := binary.LittleEndian.Uint16(rawBytes[i*2:])
			out[i] = fp16ToFloat32(bits)
		}
	case "BF16":
		for i := 0; i < nelem; i++ {
			bits := binary.LittleEndian.Uint16(rawBytes[i*2:])
			out[i] = bf16ToFloat32(bits)
		}
	default:
		return nil, nil, fmt.Errorf("unsupported dtype %q for float32 conversion", ti.Dtype)
	}

	return out, ti, nil
}

// ReadTensorI32 reads a tensor and returns it as []int32.
func (st *SafeTensors) ReadTensorI32(name string) ([]int32, *TensorInfo, error) {
	ti, err := st.GetInfo(name)
	if err != nil {
		return nil, nil, err
	}

	shardFile := st.weightMap[name]
	si := st.shards[shardFile]

	f, err := os.Open(si.path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	dataStart := int64(8 + si.headerSize + ti.DataOffsets[0])
	dataLen := ti.DataOffsets[1] - ti.DataOffsets[0]

	if _, err := f.Seek(dataStart, io.SeekStart); err != nil {
		return nil, nil, fmt.Errorf("seek: %w", err)
	}

	rawBytes := make([]byte, dataLen)
	if _, err := io.ReadFull(f, rawBytes); err != nil {
		return nil, nil, fmt.Errorf("read tensor data: %w", err)
	}

	nelem := ti.NumElements()
	out := make([]int32, nelem)
	for i := 0; i < nelem; i++ {
		out[i] = int32(binary.LittleEndian.Uint32(rawBytes[i*4:]))
	}
	return out, ti, nil
}

// DequantAWQ reads AWQ-quantized weights (qweight + qzeros + scales) and
// returns the dequantized float32 weight matrix.
// AWQ GEMM packing: 8 int4 values packed per int32 along the output dimension.
//
// Tensor shapes:
//   qweight: [inFeatures, outFeatures/8]  — 8 int4 output values packed per int32
//   qzeros:  [numGroups, outFeatures/8]   — 8 int4 zero points packed per int32
//   scales:  [numGroups, outFeatures]      — fp16 scale per group per output
//
// Dequant: weight[row][col] = (qweight_int4 - qzeros_int4) * scale
// Returns: [outFeatures, inFeatures] row-major (standard weight layout for matVec)
func (st *SafeTensors) DequantAWQ(prefix string, groupSize int) ([]float32, int, int, error) {
	// Read packed quantized weights [inFeatures, outFeatures/8]
	qweight, qwInfo, err := st.ReadTensorI32(prefix + "qweight")
	if err != nil {
		return nil, 0, 0, fmt.Errorf("qweight: %w", err)
	}

	// Read zero points [numGroups, outFeatures/8]
	qzeros, _, err := st.ReadTensorI32(prefix + "qzeros")
	if err != nil {
		return nil, 0, 0, fmt.Errorf("qzeros: %w", err)
	}

	// Read scales [numGroups, outFeatures]
	scales, scInfo, err := st.ReadTensorFloat32(prefix + "scales")
	if err != nil {
		return nil, 0, 0, fmt.Errorf("scales: %w", err)
	}

	inFeatures := qwInfo.Shape[0]
	packedCols := qwInfo.Shape[1]    // outFeatures / 8
	outFeatures := packedCols * 8
	numGroups := scInfo.Shape[0]
	_ = numGroups

	// Output: [outFeatures, inFeatures] row-major
	out := make([]float32, outFeatures*inFeatures)

	for row := 0; row < inFeatures; row++ {
		group := row / groupSize

		for packedCol := 0; packedCol < packedCols; packedCol++ {
			packed := uint32(qweight[row*packedCols+packedCol])

			// Unpack 8 int4 values from this int32
			zeroPacked := uint32(qzeros[group*packedCols+packedCol])

			for k := 0; k < 8; k++ {
				col := packedCol*8 + k
				val := int32((packed >> (4 * k)) & 0xF)
				zero := int32((zeroPacked >> (4 * k)) & 0xF)
				scale := scales[group*outFeatures+col]

				// Store as [outFeatures, inFeatures] for matVec: out[col] += W[col][row] * x[row]
				out[col*inFeatures+row] = float32(val-zero) * scale
			}
		}
	}

	return out, outFeatures, inFeatures, nil
}

// HasTensor returns true if the given tensor name exists.
func (st *SafeTensors) HasTensor(name string) bool {
	_, ok := st.weightMap[name]
	return ok
}

// ReadTensorRaw reads raw bytes for a tensor (no conversion).
func (st *SafeTensors) ReadTensorRaw(name string) ([]byte, *TensorInfo, error) {
	ti, err := st.GetInfo(name)
	if err != nil {
		return nil, nil, err
	}

	shardFile := st.weightMap[name]
	si := st.shards[shardFile]

	f, err := os.Open(si.path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	dataStart := int64(8 + si.headerSize + ti.DataOffsets[0])
	dataLen := ti.DataOffsets[1] - ti.DataOffsets[0]

	if _, err := f.Seek(dataStart, io.SeekStart); err != nil {
		return nil, nil, err
	}

	data := make([]byte, dataLen)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, nil, err
	}

	return data, ti, nil
}

// ListTensors returns all tensor names matching a prefix.
func (st *SafeTensors) ListTensors(prefix string) []string {
	var matches []string
	for _, name := range st.TensorNames {
		if strings.HasPrefix(name, prefix) {
			matches = append(matches, name)
		}
	}
	return matches
}

// fp16 to float32 conversion
func fp16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	frac := uint32(bits) & 0x3FF

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
	}
	if exp == 0x1F {
		if frac == 0 {
			return math.Float32frombits((sign << 31) | (0xFF << 23)) // inf
		}
		return math.Float32frombits((sign << 31) | (0xFF << 23) | (frac << 13)) // nan
	}
	return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

// bf16 to float32 conversion (just shift left by 16 bits)
func bf16ToFloat32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}

// === SafeTensors Writer ===

// SaveSafeTensors writes tensors to a single .safetensors file.
// tensors is a map of name → {data []float32, shape []int}.
func SaveSafeTensors(path string, tensors map[string]SaveTensor) error {
	// Sort tensor names for deterministic output
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	// Build header JSON and compute data offsets
	type headerEntry struct {
		Dtype       string    `json:"dtype"`
		Shape       []int     `json:"shape"`
		DataOffsets [2]uint64 `json:"data_offsets"`
	}
	header := make(map[string]headerEntry)
	var offset uint64
	var dataChunks [][]byte

	for _, name := range names {
		t := tensors[name]
		byteLen := uint64(len(t.Data) * 4) // float32 = 4 bytes
		header[name] = headerEntry{
			Dtype:       "F32",
			Shape:       t.Shape,
			DataOffsets: [2]uint64{offset, offset + byteLen},
		}
		// Convert float32 → little-endian bytes
		buf := make([]byte, byteLen)
		for i, v := range t.Data {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		dataChunks = append(dataChunks, buf)
		offset += byteLen
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("marshal header: %w", err)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Write 8-byte header size
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		return fmt.Errorf("write header size: %w", err)
	}
	// Write header JSON
	if _, err := f.Write(headerJSON); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	// Write tensor data
	for _, chunk := range dataChunks {
		if _, err := f.Write(chunk); err != nil {
			return fmt.Errorf("write data: %w", err)
		}
	}

	return nil
}

// SaveTensor holds data for writing a tensor to safetensors.
type SaveTensor struct {
	Data  []float32
	Shape []int
}

// QuantizedTensor holds INT8-quantized weight data with per-row absmax scales.
// Memory layout: each row quantized independently.
// Dequant: float_val = int8_val * (scale / 127.0)
type QuantizedTensor struct {
	DataInt8 []int8     // [rows * cols] quantized weights
	Scales   []float32  // [rows] per-row absmax scale factors
	Shape    []int      // original shape [rows, cols]
	Rows     int
	Cols     int
}

// QuantizeToInt8 quantizes FP32 data to INT8 with per-row absmax scaling.
func QuantizeToInt8(data []float32, rows, cols int) *QuantizedTensor {
	scales := make([]float32, rows)
	dataI8 := make([]int8, rows*cols)
	for r := 0; r < rows; r++ {
		var absmax float32
		for j := 0; j < cols; j++ {
			v := data[r*cols+j]
			if v < 0 { v = -v }
			if v > absmax { absmax = v }
		}
		scales[r] = absmax
		if absmax == 0 { continue }
		invScale := 127.0 / float64(absmax)
		for j := 0; j < cols; j++ {
			q := int32(math.Round(float64(data[r*cols+j]) * invScale))
			if q > 127 { q = 127 } else if q < -127 { q = -127 }
			dataI8[r*cols+j] = int8(q)
		}
	}
	return &QuantizedTensor{DataInt8: dataI8, Scales: scales, Shape: []int{rows, cols}, Rows: rows, Cols: cols}
}

// ReadTensorInt8 reads a tensor from safetensors and quantizes it to INT8
// with per-row absmax scaling. This compresses weights 4x (FP32→INT8) or
// 2x (BF16→INT8) while preserving most of the information.
//
// For 14B params: FP32=56GB, BF16=28GB, INT8=14GB — fits in 32GB VRAM.
func (st *SafeTensors) ReadTensorInt8(name string) (*QuantizedTensor, error) {
	ti, err := st.GetInfo(name)
	if err != nil {
		return nil, err
	}

	shardFile := st.weightMap[name]
	si := st.shards[shardFile]

	f, err := os.Open(si.path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dataStart := int64(8 + si.headerSize + ti.DataOffsets[0])
	dataLen := ti.DataOffsets[1] - ti.DataOffsets[0]
	if _, err := f.Seek(dataStart, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}
	rawBytes := make([]byte, dataLen)
	if _, err := io.ReadFull(f, rawBytes); err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}

	nelem := ti.NumElements()
	if len(ti.Shape) < 2 {
		return nil, fmt.Errorf("INT8 quant requires 2D tensor, got shape %v", ti.Shape)
	}
	rows := ti.Shape[0]
	cols := nelem / rows

	// First pass: convert to float32 and find per-row absmax
	scales := make([]float32, rows)
	dataI8 := make([]int8, nelem)

	// Process row by row to minimize memory: convert chunk → quantize → discard
	for r := 0; r < rows; r++ {
		rowStart := r * cols
		var absmax float32

		// Find absmax for this row
		switch ti.Dtype {
		case "F32":
			for j := 0; j < cols; j++ {
				idx := (rowStart + j) * 4
				bits := binary.LittleEndian.Uint32(rawBytes[idx:])
				v := math.Float32frombits(bits)
				if v < 0 {
					v = -v
				}
				if v > absmax {
					absmax = v
				}
			}
		case "F16":
			for j := 0; j < cols; j++ {
				idx := (rowStart + j) * 2
				bits := binary.LittleEndian.Uint16(rawBytes[idx:])
				v := fp16ToFloat32(bits)
				if v < 0 {
					v = -v
				}
				if v > absmax {
					absmax = v
				}
			}
		case "BF16":
			for j := 0; j < cols; j++ {
				idx := (rowStart + j) * 2
				bits := binary.LittleEndian.Uint16(rawBytes[idx:])
				v := bf16ToFloat32(bits)
				if v < 0 {
					v = -v
				}
				if v > absmax {
					absmax = v
				}
			}
		default:
			return nil, fmt.Errorf("unsupported dtype %q for INT8 quantization", ti.Dtype)
		}

		scales[r] = absmax
		if absmax == 0 {
			// Zero row — all zeros
			continue
		}

		// Quantize: round(val / absmax * 127) clamped to [-127, 127]
		invScale := 127.0 / float64(absmax)
		switch ti.Dtype {
		case "F32":
			for j := 0; j < cols; j++ {
				idx := (rowStart + j) * 4
				bits := binary.LittleEndian.Uint32(rawBytes[idx:])
				v := float64(math.Float32frombits(bits))
				q := int32(math.Round(v * invScale))
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				dataI8[rowStart+j] = int8(q)
			}
		case "F16":
			for j := 0; j < cols; j++ {
				idx := (rowStart + j) * 2
				bits := binary.LittleEndian.Uint16(rawBytes[idx:])
				v := float64(fp16ToFloat32(bits))
				q := int32(math.Round(v * invScale))
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				dataI8[rowStart+j] = int8(q)
			}
		case "BF16":
			for j := 0; j < cols; j++ {
				idx := (rowStart + j) * 2
				bits := binary.LittleEndian.Uint16(rawBytes[idx:])
				v := float64(bf16ToFloat32(bits))
				q := int32(math.Round(v * invScale))
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				dataI8[rowStart+j] = int8(q)
			}
		}
	}

	return &QuantizedTensor{
		DataInt8: dataI8,
		Scales:   scales,
		Shape:    ti.Shape,
		Rows:     rows,
		Cols:     cols,
	}, nil
}
