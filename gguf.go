package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// GGUF writer — produces files compatible with llama.cpp and Ollama.

const (
	ggufMagic     = 0x46554747 // "GGUF" in little-endian
	ggufVersion   = 3
	ggufAlignment = 32

	// Value types
	ggufTypeUint32  = 4
	ggufTypeFloat32 = 6
	ggufTypeString  = 8
	ggufTypeArray   = 9

	// Tensor data types
	GGUFTypeF32 = 0
	GGUFTypeF16 = 1
	GGUFTypeQ8  = 8
)

// GGUFWriter builds and writes a GGUF file.
type GGUFWriter struct {
	metadata []ggufKV
	tensors  []ggufTensorDesc
	data     [][]byte // raw tensor data in order
}

type ggufKV struct {
	key       string
	valueType uint32
	value     interface{} // string, uint32, float32, or []string/[]float32/[]int32
}

type ggufTensorDesc struct {
	name  string
	shape []uint64
	dtype uint32
	data  []byte
}

// NewGGUFWriter creates a new GGUF file writer.
func NewGGUFWriter() *GGUFWriter {
	return &GGUFWriter{}
}

// AddString adds a string metadata key-value pair.
func (w *GGUFWriter) AddString(key, value string) {
	w.metadata = append(w.metadata, ggufKV{key, ggufTypeString, value})
}

// AddUint32 adds a uint32 metadata key-value pair.
func (w *GGUFWriter) AddUint32(key string, value uint32) {
	w.metadata = append(w.metadata, ggufKV{key, ggufTypeUint32, value})
}

// AddFloat32 adds a float32 metadata key-value pair.
func (w *GGUFWriter) AddFloat32(key string, value float32) {
	w.metadata = append(w.metadata, ggufKV{key, ggufTypeFloat32, value})
}

// AddStringArray adds a string array metadata value.
func (w *GGUFWriter) AddStringArray(key string, values []string) {
	w.metadata = append(w.metadata, ggufKV{key, ggufTypeArray, values})
}

// AddFloat32Array adds a float32 array metadata value.
func (w *GGUFWriter) AddFloat32Array(key string, values []float32) {
	w.metadata = append(w.metadata, ggufKV{key, ggufTypeArray, values})
}

// AddInt32Array adds an int32 array metadata value.
func (w *GGUFWriter) AddInt32Array(key string, values []int32) {
	w.metadata = append(w.metadata, ggufKV{key, ggufTypeArray, values})
}

// AddTensorF32 adds a float32 tensor.
func (w *GGUFWriter) AddTensorF32(name string, data []float32, shape ...int) {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	dims := make([]uint64, len(shape))
	for i, s := range shape { dims[i] = uint64(s) }
	w.tensors = append(w.tensors, ggufTensorDesc{name, dims, GGUFTypeF32, buf})
}

// AddTensorQ8_0 quantizes float32 data to Q8_0 (8-bit) and adds it.
// Q8_0: blocks of 32 elements. Per block: float16 scale + 32 × int8.
// 34 bytes per block. ~4.25x compression vs F32.
func (w *GGUFWriter) AddTensorQ8_0(name string, data []float32, shape ...int) {
	const blockSize = 32
	nBlocks := (len(data) + blockSize - 1) / blockSize
	buf := make([]byte, nBlocks*34) // 34 bytes per block (2 + 32)

	for b := 0; b < nBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}
		block := data[start:end]

		// Find max absolute value for scale
		var absMax float32
		for _, v := range block {
			if v < 0 && -v > absMax {
				absMax = -v
			} else if v > absMax {
				absMax = v
			}
		}

		scale := absMax / 127.0
		if scale == 0 {
			scale = 1.0 // avoid division by zero
		}
		invScale := 127.0 / absMax
		if absMax == 0 {
			invScale = 0
		}

		// Write scale as float16
		off := b * 34
		scaleFP16 := float32ToFP16(scale)
		binary.LittleEndian.PutUint16(buf[off:], scaleFP16)

		// Quantize to int8
		for i, v := range block {
			q := int(math.Round(float64(v * invScale)))
			if q > 127 {
				q = 127
			} else if q < -128 {
				q = -128
			}
			buf[off+2+i] = byte(int8(q))
		}
		// Zero-pad if block is not full
		for i := len(block); i < blockSize; i++ {
			buf[off+2+i] = 0
		}
	}

	dims := make([]uint64, len(shape))
	for i, s := range shape {
		dims[i] = uint64(s)
	}
	w.tensors = append(w.tensors, ggufTensorDesc{name, dims, GGUFTypeQ8, buf})
}

// AddTensorQ4_0 quantizes float32 data to Q4_0 (4-bit) and adds it.
// Q4_0: blocks of 32 elements. Per block: float16 scale + 16 bytes (32 × 4-bit).
// 18 bytes per block. ~8.9x compression vs F32.
func (w *GGUFWriter) AddTensorQ4_0(name string, data []float32, shape ...int) {
	const blockSize = 32
	nBlocks := (len(data) + blockSize - 1) / blockSize
	buf := make([]byte, nBlocks*18) // 18 bytes per block (2 + 16)

	for b := 0; b < nBlocks; b++ {
		start := b * blockSize
		end := start + blockSize
		if end > len(data) {
			end = len(data)
		}
		block := data[start:end]

		var absMax float32
		for _, v := range block {
			if v < 0 && -v > absMax {
				absMax = -v
			} else if v > absMax {
				absMax = v
			}
		}

		scale := absMax / 7.0 // 4-bit signed: -8..7, use 7 for symmetric
		if scale == 0 {
			scale = 1.0
		}
		invScale := 7.0 / absMax
		if absMax == 0 {
			invScale = 0
		}

		off := b * 18
		binary.LittleEndian.PutUint16(buf[off:], float32ToFP16(scale))

		// Pack 32 values into 16 bytes (2 per byte, low nibble first)
		for i := 0; i < blockSize/2; i++ {
			var lo, hi int
			if i*2 < len(block) {
				lo = int(math.Round(float64(block[i*2]*invScale))) + 8
			} else {
				lo = 8
			}
			if i*2+1 < len(block) {
				hi = int(math.Round(float64(block[i*2+1]*invScale))) + 8
			} else {
				hi = 8
			}
			if lo < 0 { lo = 0 } else if lo > 15 { lo = 15 }
			if hi < 0 { hi = 0 } else if hi > 15 { hi = 15 }
			buf[off+2+i] = byte(lo) | byte(hi<<4)
		}
	}

	dims := make([]uint64, len(shape))
	for i, s := range shape {
		dims[i] = uint64(s)
	}
	w.tensors = append(w.tensors, ggufTensorDesc{name, dims, 2, buf}) // type 2 = Q4_0
}

// float32ToFP16 converts a float32 to IEEE 754 half-precision (float16).
func float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	frac := bits & 0x007FFFFF
	if exp <= 0 {
		return sign
	}
	if exp >= 31 {
		return sign | 0x7C00
	}
	return sign | uint16(exp<<10) | uint16(frac>>13)
}

// Write writes the GGUF file to the given path.
func (w *GGUFWriter) Write(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header
	binary.Write(f, binary.LittleEndian, uint32(ggufMagic))
	binary.Write(f, binary.LittleEndian, uint32(ggufVersion))
	binary.Write(f, binary.LittleEndian, uint64(len(w.tensors)))
	binary.Write(f, binary.LittleEndian, uint64(len(w.metadata)))

	// Metadata
	for _, kv := range w.metadata {
		writeGGUFString(f, kv.key)
		binary.Write(f, binary.LittleEndian, kv.valueType)
		switch v := kv.value.(type) {
		case string:
			writeGGUFString(f, v)
		case uint32:
			binary.Write(f, binary.LittleEndian, v)
		case float32:
			binary.Write(f, binary.LittleEndian, v)
		case []string:
			binary.Write(f, binary.LittleEndian, uint32(ggufTypeString))
			binary.Write(f, binary.LittleEndian, uint64(len(v)))
			for _, s := range v { writeGGUFString(f, s) }
		case []float32:
			binary.Write(f, binary.LittleEndian, uint32(ggufTypeFloat32))
			binary.Write(f, binary.LittleEndian, uint64(len(v)))
			for _, x := range v { binary.Write(f, binary.LittleEndian, x) }
		case []int32:
			binary.Write(f, binary.LittleEndian, uint32(5)) // INT32
			binary.Write(f, binary.LittleEndian, uint64(len(v)))
			for _, x := range v { binary.Write(f, binary.LittleEndian, x) }
		}
	}

	// Tensor descriptors
	offset := uint64(0)
	for _, t := range w.tensors {
		writeGGUFString(f, t.name)
		binary.Write(f, binary.LittleEndian, uint32(len(t.shape)))
		for _, d := range t.shape {
			binary.Write(f, binary.LittleEndian, d)
		}
		binary.Write(f, binary.LittleEndian, t.dtype)
		binary.Write(f, binary.LittleEndian, offset)
		// Align next tensor
		sz := uint64(len(t.data))
		offset += sz
		if offset%ggufAlignment != 0 {
			offset += ggufAlignment - (offset % ggufAlignment)
		}
	}

	// Pad to alignment before tensor data
	pos, _ := f.Seek(0, 1)
	if pad := ggufAlignment - (int(pos) % ggufAlignment); pad != ggufAlignment {
		f.Write(make([]byte, pad))
	}

	// Tensor data
	for _, t := range w.tensors {
		f.Write(t.data)
		// Pad to alignment
		if pad := ggufAlignment - (len(t.data) % ggufAlignment); pad != ggufAlignment {
			f.Write(make([]byte, pad))
		}
	}

	return nil
}

func writeGGUFString(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.Write([]byte(s))
}

// ConvertSafetensorsToGGUF converts a safetensors model to GGUF format.
// quantType: "f32", "q8_0", "q4_0"
func ConvertSafetensorsToGGUF(modelDir, outputPath string, quantType ...string) error {
	qtype := "f32"
	if len(quantType) > 0 && quantType[0] != "" {
		qtype = quantType[0]
	}
	st, err := OpenSafeTensors(modelDir)
	if err != nil {
		return fmt.Errorf("open model: %w", err)
	}

	// Read config
	configData, err := os.ReadFile(modelDir + "/config.json")
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}

	// Parse config (simple JSON extraction)
	getInt := func(key string) int {
		// Quick and dirty JSON int extraction
		import_json := false
		_ = import_json
		for i := 0; i < len(configData)-len(key)-5; i++ {
			if string(configData[i:i+len(key)]) == key {
				// Find the colon then the number
				for j := i + len(key); j < len(configData); j++ {
					if configData[j] >= '0' && configData[j] <= '9' {
						n := 0
						for ; j < len(configData) && configData[j] >= '0' && configData[j] <= '9'; j++ {
							n = n*10 + int(configData[j]-'0')
						}
						return n
					}
				}
			}
		}
		return 0
	}
	getFloat := func(key string) float64 {
		for i := 0; i < len(configData)-len(key)-5; i++ {
			if string(configData[i:i+len(key)]) == key {
				for j := i + len(key); j < len(configData); j++ {
					if (configData[j] >= '0' && configData[j] <= '9') || configData[j] == '-' {
						end := j
						for ; end < len(configData) && configData[end] != ',' && configData[end] != '}'; end++ {}
						var v float64
						fmt.Sscanf(string(configData[j:end]), "%f", &v)
						return v
					}
				}
			}
		}
		return 0
	}

	dim := getInt("hidden_size")
	layers := getInt("num_hidden_layers")
	heads := getInt("num_attention_heads")
	kvHeads := getInt("num_key_value_heads")
	ffnDim := getInt("intermediate_size")
	vocabSize := getInt("vocab_size")
	maxSeq := getInt("max_position_embeddings")
	ropeTheta := getFloat("rope_theta")
	if kvHeads == 0 { kvHeads = heads }
	if maxSeq == 0 { maxSeq = 512 }
	if ropeTheta == 0 { ropeTheta = 10000 }

	w := NewGGUFWriter()

	// Metadata
	arch := "llama" // Qwen2-compatible uses llama architecture in GGUF
	w.AddString("general.architecture", arch)
	w.AddString("general.name", "mongoose-trained")
	w.AddUint32(arch+".context_length", uint32(maxSeq))
	w.AddUint32(arch+".embedding_length", uint32(dim))
	w.AddUint32(arch+".block_count", uint32(layers))
	w.AddUint32(arch+".feed_forward_length", uint32(ffnDim))
	w.AddUint32(arch+".attention.head_count", uint32(heads))
	w.AddUint32(arch+".attention.head_count_kv", uint32(kvHeads))
	w.AddFloat32(arch+".rope.freq_base", float32(ropeTheta))
	w.AddFloat32(arch+".attention.layer_norm_rms_epsilon", 1e-6)
	// File type: 0=F32, 7=Q8_0, 2=Q4_0
	fileType := uint32(0)
	switch qtype {
	case "q8_0": fileType = 7
	case "q4_0": fileType = 2
	}
	w.AddUint32("general.file_type", fileType)

	// Tokenizer — byte-level
	tokens := make([]string, vocabSize)
	scores := make([]float32, vocabSize)
	types := make([]int32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		if i < 256 {
			tokens[i] = string([]byte{byte(i)})
		} else {
			tokens[i] = fmt.Sprintf("[%d]", i)
		}
		types[i] = 1 // normal token
	}
	w.AddString("tokenizer.ggml.model", "llama")
	w.AddStringArray("tokenizer.ggml.tokens", tokens)
	w.AddFloat32Array("tokenizer.ggml.scores", scores)
	w.AddInt32Array("tokenizer.ggml.token_type", types)

	// Tensors — map mongoose names to GGUF names
	nameMap := map[string]string{
		"model.embed_tokens.weight":    "token_embd.weight",
		"model.norm.weight":            "output_norm.weight",
	}
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := fmt.Sprintf("blk.%d.", l)
		nameMap[p+"input_layernorm.weight"] = g + "attn_norm.weight"
		nameMap[p+"self_attn.q_proj.weight"] = g + "attn_q.weight"
		nameMap[p+"self_attn.k_proj.weight"] = g + "attn_k.weight"
		nameMap[p+"self_attn.v_proj.weight"] = g + "attn_v.weight"
		nameMap[p+"self_attn.o_proj.weight"] = g + "attn_output.weight"
		nameMap[p+"self_attn.q_proj.bias"] = g + "attn_q.bias"
		nameMap[p+"self_attn.k_proj.bias"] = g + "attn_k.bias"
		nameMap[p+"self_attn.v_proj.bias"] = g + "attn_v.bias"
		nameMap[p+"post_attention_layernorm.weight"] = g + "ffn_norm.weight"
		nameMap[p+"mlp.gate_proj.weight"] = g + "ffn_gate.weight"
		nameMap[p+"mlp.up_proj.weight"] = g + "ffn_up.weight"
		nameMap[p+"mlp.down_proj.weight"] = g + "ffn_down.weight"
	}
	// output.weight = embed (tied)
	nameMap["lm_head.weight"] = "output.weight"

	// addTensor picks F32 or quantized based on tensor type and qtype setting.
	// Norms, biases, and embeddings stay F32. Weight matrices get quantized.
	isWeight := func(name string) bool {
		// Weight matrices: anything with "weight" that's not a norm or embedding
		for _, skip := range []string{"norm", "embd", "token_embd", "output.weight"} {
			if len(name) >= len(skip) && name[len(name)-len(skip):] == skip ||
				(len(name) > len(skip) && name[:len(skip)] == skip) {
				return false
			}
		}
		return true
	}
	addTensor := func(ggufName string, data []float32, shape []int) {
		if qtype != "f32" && isWeight(ggufName) && len(data) >= 32 {
			switch qtype {
			case "q8_0":
				w.AddTensorQ8_0(ggufName, data, shape...)
			case "q4_0":
				w.AddTensorQ4_0(ggufName, data, shape...)
			default:
				w.AddTensorF32(ggufName, data, shape...)
			}
		} else {
			w.AddTensorF32(ggufName, data, shape...)
		}
	}

	for stName, ggufName := range nameMap {
		data, info, err := st.ReadTensorFloat32(stName)
		if err != nil {
			continue // optional tensors (biases)
		}
		addTensor(ggufName, data, info.Shape)
	}

	// If no separate lm_head, use embed (tied weights)
	if !st.HasTensor("lm_head.weight") {
		data, info, _ := st.ReadTensorFloat32("model.embed_tokens.weight")
		if data != nil {
			addTensor("output.weight", data, info.Shape)
		}
	}

	return w.Write(outputPath)
}

// GGUFReader reads tensors from a GGUF file.
type GGUFReader struct {
	f        *os.File
	metadata map[string]interface{}
	tensors  map[string]ggufTensorInfo
	dataOff  int64
}

type ggufTensorInfo struct {
	name   string
	shape  []int
	dtype  uint32
	offset uint64
	nElems int
}

// OpenGGUF opens a GGUF file for reading.
func OpenGGUF(path string) (*GGUFReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	var magic, version uint32
	binary.Read(f, binary.LittleEndian, &magic)
	binary.Read(f, binary.LittleEndian, &version)
	if magic != ggufMagic {
		f.Close()
		return nil, fmt.Errorf("not a GGUF file (magic %x)", magic)
	}
	if version < 2 || version > 3 {
		f.Close()
		return nil, fmt.Errorf("unsupported GGUF version %d", version)
	}

	var tensorCount, metadataCount uint64
	binary.Read(f, binary.LittleEndian, &tensorCount)
	binary.Read(f, binary.LittleEndian, &metadataCount)

	r := &GGUFReader{
		f:        f,
		metadata: make(map[string]interface{}),
		tensors:  make(map[string]ggufTensorInfo),
	}

	for i := uint64(0); i < metadataCount; i++ {
		key := readGGUFString(f)
		var valueType uint32
		binary.Read(f, binary.LittleEndian, &valueType)
		value := readGGUFValue(f, valueType)
		r.metadata[key] = value
	}

	for i := uint64(0); i < tensorCount; i++ {
		name := readGGUFString(f)
		var nDims uint32
		binary.Read(f, binary.LittleEndian, &nDims)
		shape := make([]int, nDims)
		nElems := 1
		for d := uint32(0); d < nDims; d++ {
			var dim uint64
			binary.Read(f, binary.LittleEndian, &dim)
			shape[d] = int(dim)
			nElems *= int(dim)
		}
		var dtype uint32
		var offset uint64
		binary.Read(f, binary.LittleEndian, &dtype)
		binary.Read(f, binary.LittleEndian, &offset)
		r.tensors[name] = ggufTensorInfo{name: name, shape: shape, dtype: dtype, offset: offset, nElems: nElems}
	}

	// Data starts at next alignment boundary
	pos, _ := f.Seek(0, 1)
	if rem := pos % ggufAlignment; rem != 0 {
		pos += int64(ggufAlignment) - rem
	}
	r.dataOff = pos

	return r, nil
}

func (r *GGUFReader) Close() error { return r.f.Close() }

// Metadata returns all metadata key-value pairs.
func (r *GGUFReader) Metadata() map[string]interface{} { return r.metadata }

// MetadataString returns a string metadata value.
func (r *GGUFReader) MetadataString(key string) string {
	if v, ok := r.metadata[key].(string); ok {
		return v
	}
	return ""
}

// MetadataUint32 returns a uint32 metadata value.
func (r *GGUFReader) MetadataUint32(key string) uint32 {
	if v, ok := r.metadata[key].(uint32); ok {
		return v
	}
	return 0
}

// MetadataFloat32 returns a float32 metadata value.
func (r *GGUFReader) MetadataFloat32(key string) float32 {
	if v, ok := r.metadata[key].(float32); ok {
		return v
	}
	return 0
}

// TensorNames returns all tensor names.
func (r *GGUFReader) TensorNames() []string {
	names := make([]string, 0, len(r.tensors))
	for n := range r.tensors {
		names = append(names, n)
	}
	return names
}

// HasTensor returns true if the tensor exists.
func (r *GGUFReader) HasTensor(name string) bool {
	_, ok := r.tensors[name]
	return ok
}

// TensorShape returns the shape of a tensor.
func (r *GGUFReader) TensorShape(name string) []int {
	if t, ok := r.tensors[name]; ok {
		return t.shape
	}
	return nil
}

// ReadTensorRaw reads raw bytes for a tensor without dequantization.
// Returns the raw bytes, dtype code, element count, and shape.
func (r *GGUFReader) ReadTensorRaw(name string) ([]byte, uint32, int, []int, error) {
	t, ok := r.tensors[name]
	if !ok {
		return nil, 0, 0, nil, fmt.Errorf("tensor %q not found", name)
	}
	var nBytes int
	switch t.dtype {
	case GGUFTypeF32:
		nBytes = t.nElems * 4
	case GGUFTypeF16:
		nBytes = t.nElems * 2
	case GGUFTypeQ8: // Q8_0: 34 bytes per 32 elements
		nBytes = ((t.nElems + 31) / 32) * 34
	case 2: // Q4_0: 18 bytes per 32 elements
		nBytes = ((t.nElems + 31) / 32) * 18
	default:
		return nil, 0, 0, nil, fmt.Errorf("tensor %q: unsupported dtype %d", name, t.dtype)
	}
	r.f.Seek(r.dataOff+int64(t.offset), 0)
	data := make([]byte, nBytes)
	r.f.Read(data)
	return data, t.dtype, t.nElems, t.shape, nil
}

// TensorDtype returns the GGUF dtype code for a tensor.
func (r *GGUFReader) TensorDtype(name string) uint32 {
	if t, ok := r.tensors[name]; ok {
		return t.dtype
	}
	return 0
}

// ReadTensorFloat32 reads a tensor and dequantizes to float32.
func (r *GGUFReader) ReadTensorFloat32(name string) ([]float32, []int, error) {
	t, ok := r.tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}

	r.f.Seek(r.dataOff+int64(t.offset), 0)

	switch t.dtype {
	case GGUFTypeF32:
		data := make([]float32, t.nElems)
		binary.Read(r.f, binary.LittleEndian, data)
		return data, t.shape, nil

	case GGUFTypeF16:
		raw := make([]uint16, t.nElems)
		binary.Read(r.f, binary.LittleEndian, raw)
		data := make([]float32, t.nElems)
		for i, v := range raw {
			data[i] = fp16ToFloat32(v)
		}
		return data, t.shape, nil

	case GGUFTypeQ8: // Q8_0
		return r.dequantQ8(t)

	case 2: // Q4_0
		return r.dequantQ4(t)

	default:
		return nil, nil, fmt.Errorf("tensor %q: unsupported dtype %d", name, t.dtype)
	}
}

func (r *GGUFReader) dequantQ8(t ggufTensorInfo) ([]float32, []int, error) {
	const blockSize = 32
	nBlocks := (t.nElems + blockSize - 1) / blockSize
	raw := make([]byte, nBlocks*34)
	r.f.Read(raw)

	data := make([]float32, t.nElems)
	for b := 0; b < nBlocks; b++ {
		off := b * 34
		scaleFP16 := binary.LittleEndian.Uint16(raw[off:])
		scale := fp16ToFloat32(scaleFP16)
		count := blockSize
		if b*blockSize+count > t.nElems {
			count = t.nElems - b*blockSize
		}
		for i := 0; i < count; i++ {
			data[b*blockSize+i] = float32(int8(raw[off+2+i])) * scale
		}
	}
	return data, t.shape, nil
}

func (r *GGUFReader) dequantQ4(t ggufTensorInfo) ([]float32, []int, error) {
	const blockSize = 32
	nBlocks := (t.nElems + blockSize - 1) / blockSize
	raw := make([]byte, nBlocks*18)
	r.f.Read(raw)

	data := make([]float32, t.nElems)
	for b := 0; b < nBlocks; b++ {
		off := b * 18
		scaleFP16 := binary.LittleEndian.Uint16(raw[off:])
		scale := fp16ToFloat32(scaleFP16)
		count := blockSize
		if b*blockSize+count > t.nElems {
			count = t.nElems - b*blockSize
		}
		for i := 0; i < count/2; i++ {
			packed := raw[off+2+i]
			lo := int(packed&0x0F) - 8
			hi := int(packed>>4) - 8
			if i*2 < count {
				data[b*blockSize+i*2] = float32(lo) * scale
			}
			if i*2+1 < count {
				data[b*blockSize+i*2+1] = float32(hi) * scale
			}
		}
	}
	return data, t.shape, nil
}

func readGGUFString(f *os.File) string {
	var n uint64
	binary.Read(f, binary.LittleEndian, &n)
	buf := make([]byte, n)
	f.Read(buf)
	return string(buf)
}

func readGGUFValue(f *os.File, valueType uint32) interface{} {
	switch valueType {
	case ggufTypeUint32:
		var v uint32
		binary.Read(f, binary.LittleEndian, &v)
		return v
	case ggufTypeFloat32:
		var v float32
		binary.Read(f, binary.LittleEndian, &v)
		return v
	case ggufTypeString:
		return readGGUFString(f)
	case ggufTypeArray:
		var elemType uint32
		var count uint64
		binary.Read(f, binary.LittleEndian, &elemType)
		binary.Read(f, binary.LittleEndian, &count)
		for i := uint64(0); i < count; i++ {
			readGGUFValue(f, elemType)
		}
		return nil // arrays stored in metadata but not returned for simplicity
	default:
		// Skip unknown types: read 4 bytes
		var v uint32
		binary.Read(f, binary.LittleEndian, &v)
		return v
	}
}

// fp16ToFloat32 is defined in safetensors.go
