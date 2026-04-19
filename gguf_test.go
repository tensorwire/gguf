package gguf

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSafeTensorsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	original := map[string]SaveTensor{
		"weight": {Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape: []int{2, 3}},
		"bias":   {Data: []float32{0.1, 0.2, 0.3}, Shape: []int{3}},
	}

	if err := SaveSafeTensors(path, original); err != nil {
		t.Fatalf("SaveSafeTensors: %v", err)
	}

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("OpenSafeTensors: %v", err)
	}

	if !st.HasTensor("weight") {
		t.Fatal("missing tensor: weight")
	}
	if !st.HasTensor("bias") {
		t.Fatal("missing tensor: bias")
	}
	if st.HasTensor("nonexistent") {
		t.Fatal("HasTensor returned true for nonexistent")
	}

	data, info, err := st.ReadTensorFloat32("weight")
	if err != nil {
		t.Fatalf("ReadTensorFloat32(weight): %v", err)
	}
	if info.NumElements() != 6 {
		t.Fatalf("weight elements: got %d, want 6", info.NumElements())
	}
	for i, want := range original["weight"].Data {
		if data[i] != want {
			t.Errorf("weight[%d] = %f, want %f", i, data[i], want)
		}
	}

	biasData, biasInfo, err := st.ReadTensorFloat32("bias")
	if err != nil {
		t.Fatalf("ReadTensorFloat32(bias): %v", err)
	}
	if biasInfo.NumElements() != 3 {
		t.Fatalf("bias elements: got %d, want 3", biasInfo.NumElements())
	}
	for i, want := range original["bias"].Data {
		if biasData[i] != want {
			t.Errorf("bias[%d] = %f, want %f", i, biasData[i], want)
		}
	}
}

func TestSafeTensorsListTensors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	tensors := map[string]SaveTensor{
		"model.layer.0.weight": {Data: []float32{1, 2}, Shape: []int{2}},
		"model.layer.0.bias":   {Data: []float32{3}, Shape: []int{1}},
		"model.layer.1.weight": {Data: []float32{4, 5}, Shape: []int{2}},
		"other.param":          {Data: []float32{6}, Shape: []int{1}},
	}

	if err := SaveSafeTensors(path, tensors); err != nil {
		t.Fatalf("SaveSafeTensors: %v", err)
	}

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("OpenSafeTensors: %v", err)
	}

	layer0 := st.ListTensors("model.layer.0")
	if len(layer0) != 2 {
		t.Fatalf("ListTensors(model.layer.0): got %d, want 2", len(layer0))
	}

	all := st.ListTensors("")
	if len(all) != 4 {
		t.Fatalf("ListTensors(''): got %d, want 4", len(all))
	}

	none := st.ListTensors("nonexistent")
	if len(none) != 0 {
		t.Fatalf("ListTensors(nonexistent): got %d, want 0", len(none))
	}
}

func TestQuantizeToInt8(t *testing.T) {
	data := []float32{1.0, -0.5, 0.25, -1.0, 0.0, 0.75}
	q := QuantizeToInt8(data, 2, 3)

	if q.Rows != 2 || q.Cols != 3 {
		t.Fatalf("shape: got %dx%d, want 2x3", q.Rows, q.Cols)
	}
	if len(q.Scales) != 2 {
		t.Fatalf("scales length: got %d, want 2", len(q.Scales))
	}

	if q.Scales[0] != 1.0 {
		t.Errorf("scales[0] = %f, want 1.0", q.Scales[0])
	}
	if q.Scales[1] != 1.0 {
		t.Errorf("scales[1] = %f, want 1.0", q.Scales[1])
	}

	if q.DataInt8[0] != 127 {
		t.Errorf("q[0] = %d, want 127 (max positive)", q.DataInt8[0])
	}
	if q.DataInt8[3] != -127 {
		t.Errorf("q[3] = %d, want -127 (max negative)", q.DataInt8[3])
	}
	if q.DataInt8[4] != 0 {
		t.Errorf("q[4] = %d, want 0 (zero input)", q.DataInt8[4])
	}
}

func TestQuantizeToInt8_ZeroRow(t *testing.T) {
	data := []float32{0.0, 0.0, 0.0}
	q := QuantizeToInt8(data, 1, 3)

	if q.Scales[0] != 0.0 {
		t.Errorf("zero row scale = %f, want 0.0", q.Scales[0])
	}
	for i := 0; i < 3; i++ {
		if q.DataInt8[i] != 0 {
			t.Errorf("zero row q[%d] = %d, want 0", i, q.DataInt8[i])
		}
	}
}

func TestDtypeBytes(t *testing.T) {
	cases := []struct {
		dtype string
		want  int
	}{
		{"F32", 4}, {"F16", 2}, {"BF16", 2}, {"I32", 4}, {"I8", 1},
	}
	for _, c := range cases {
		got := DtypeBytes(c.dtype)
		if got != c.want {
			t.Errorf("DtypeBytes(%q) = %d, want %d", c.dtype, got, c.want)
		}
	}
}

func TestGGUFWriteF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.gguf")

	w := NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddUint32("llama.context_length", 2048)
	w.AddTensorF32("weight", []float32{1.0, 2.0, 3.0, 4.0}, 2, 2)

	if err := w.Write(path); err != nil {
		t.Fatalf("Write: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("GGUF file is empty")
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	magic := make([]byte, 4)
	if _, err := f.Read(magic); err != nil {
		t.Fatalf("read magic: %v", err)
	}
	if string(magic) != "GGUF" {
		t.Fatalf("magic = %q, want GGUF", string(magic))
	}
}

func TestGGUFWriteQ8_0(t *testing.T) {
	dir := t.TempDir()
	pathF32 := filepath.Join(dir, "f32.gguf")
	pathQ8 := filepath.Join(dir, "q8.gguf")

	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	wF32 := NewGGUFWriter()
	wF32.AddTensorF32("w", data, 32, 32)
	if err := wF32.Write(pathF32); err != nil {
		t.Fatalf("Write F32: %v", err)
	}

	wQ8 := NewGGUFWriter()
	wQ8.AddTensorQ8_0("w", data, 32, 32)
	if err := wQ8.Write(pathQ8); err != nil {
		t.Fatalf("Write Q8: %v", err)
	}

	infoF32, _ := os.Stat(pathF32)
	infoQ8, _ := os.Stat(pathQ8)

	ratio := float64(infoF32.Size()) / float64(infoQ8.Size())
	t.Logf("F32: %d bytes, Q8_0: %d bytes, ratio: %.2fx", infoF32.Size(), infoQ8.Size(), ratio)

	if ratio < 2.0 {
		t.Errorf("Q8_0 compression ratio %.2f < 2.0x — expected ~4x", ratio)
	}
}

func TestGGUFWriteQ4_0(t *testing.T) {
	dir := t.TempDir()
	pathF32 := filepath.Join(dir, "f32.gguf")
	pathQ4 := filepath.Join(dir, "q4.gguf")

	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	wF32 := NewGGUFWriter()
	wF32.AddTensorF32("w", data, 32, 32)
	if err := wF32.Write(pathF32); err != nil {
		t.Fatalf("Write F32: %v", err)
	}

	wQ4 := NewGGUFWriter()
	wQ4.AddTensorQ4_0("w", data, 32, 32)
	if err := wQ4.Write(pathQ4); err != nil {
		t.Fatalf("Write Q4: %v", err)
	}

	infoF32, _ := os.Stat(pathF32)
	infoQ4, _ := os.Stat(pathQ4)

	ratio := float64(infoF32.Size()) / float64(infoQ4.Size())
	t.Logf("F32: %d bytes, Q4_0: %d bytes, ratio: %.2fx", infoF32.Size(), infoQ4.Size(), ratio)

	if ratio < 3.0 {
		t.Errorf("Q4_0 compression ratio %.2f < 3.0x — expected ~8x", ratio)
	}
}

func TestFloat32ToFP16RoundTrip(t *testing.T) {
	cases := []float32{0.0, 1.0, -1.0, 0.5, 3.14, -42.0, 65504.0}
	for _, v := range cases {
		bits := float32ToFP16(v)
		back := fp16ToFloat32(bits)
		diff := math.Abs(float64(v - back))
		relErr := diff / math.Max(math.Abs(float64(v)), 1e-10)
		if relErr > 0.01 {
			t.Errorf("FP16 round-trip %f → %f, relative error %.4f", v, back, relErr)
		}
	}
}

func TestFloat32ToFP16_Zero(t *testing.T) {
	bits := float32ToFP16(0.0)
	back := fp16ToFloat32(bits)
	if back != 0.0 {
		t.Errorf("FP16(0.0) = %f, want 0.0", back)
	}
}

func TestNpyRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.npy")

	original := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	if err := WriteNpy(path, original, 2, 3); err != nil {
		t.Fatalf("WriteNpy: %v", err)
	}

	arr, err := ReadNpy(path)
	if err != nil {
		t.Fatalf("ReadNpy: %v", err)
	}

	if arr.Rows() != 2 {
		t.Errorf("rows = %d, want 2", arr.Rows())
	}
	if arr.Cols() != 3 {
		t.Errorf("cols = %d, want 3", arr.Cols())
	}

	for i, want := range original {
		if arr.Data[i] != want {
			t.Errorf("data[%d] = %f, want %f", i, arr.Data[i], want)
		}
	}

	row0 := arr.Row(0)
	if len(row0) != 3 || row0[0] != 1.0 || row0[2] != 3.0 {
		t.Errorf("Row(0) = %v, want [1 2 3]", row0)
	}
}

func TestNpy1D(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test1d.npy")

	original := []float32{10.0, 20.0, 30.0}
	if err := WriteNpy(path, original, 3); err != nil {
		t.Fatalf("WriteNpy: %v", err)
	}

	arr, err := ReadNpy(path)
	if err != nil {
		t.Fatalf("ReadNpy: %v", err)
	}

	if arr.Rows() != 3 {
		t.Errorf("rows = %d, want 3", arr.Rows())
	}
	if arr.Cols() != 1 {
		t.Errorf("cols = %d, want 1", arr.Cols())
	}
}

func TestSafeTensorsGetInfo(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	tensors := map[string]SaveTensor{
		"w": {Data: []float32{1, 2, 3, 4, 5, 6}, Shape: []int{2, 3}},
	}
	if err := SaveSafeTensors(path, tensors); err != nil {
		t.Fatalf("SaveSafeTensors: %v", err)
	}

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("OpenSafeTensors: %v", err)
	}

	info, err := st.GetInfo("w")
	if err != nil {
		t.Fatalf("GetInfo: %v", err)
	}

	if info.Dtype != "F32" {
		t.Errorf("dtype = %q, want F32", info.Dtype)
	}
	if info.NumElements() != 6 {
		t.Errorf("elements = %d, want 6", info.NumElements())
	}
	if info.ByteSize() != 24 {
		t.Errorf("byte size = %d, want 24", info.ByteSize())
	}

	_, err = st.GetInfo("nonexistent")
	if err == nil {
		t.Error("GetInfo(nonexistent) should return error")
	}
}

func TestSafeTensorsReadTensorRaw(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	tensors := map[string]SaveTensor{
		"w": {Data: []float32{1.0, 2.0}, Shape: []int{2}},
	}
	if err := SaveSafeTensors(path, tensors); err != nil {
		t.Fatalf("SaveSafeTensors: %v", err)
	}

	st, err := OpenSafeTensors(path)
	if err != nil {
		t.Fatalf("OpenSafeTensors: %v", err)
	}

	raw, info, err := st.ReadTensorRaw("w")
	if err != nil {
		t.Fatalf("ReadTensorRaw: %v", err)
	}

	if len(raw) != 8 {
		t.Fatalf("raw bytes = %d, want 8 (2 floats * 4 bytes)", len(raw))
	}
	if info.NumElements() != 2 {
		t.Errorf("elements = %d, want 2", info.NumElements())
	}
}

func TestGGUFMetadata(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "meta.gguf")

	w := NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddString("general.name", "test-model")
	w.AddUint32("llama.context_length", 4096)
	w.AddFloat32("llama.rope_theta", 10000.0)
	w.AddStringArray("general.tags", []string{"test", "small"})
	w.AddFloat32Array("custom.values", []float32{1.0, 2.0, 3.0})
	w.AddInt32Array("custom.ids", []int32{100, 200})

	if err := w.Write(path); err != nil {
		t.Fatalf("Write: %v", err)
	}

	info, _ := os.Stat(path)
	if info.Size() < 50 {
		t.Errorf("file too small: %d bytes", info.Size())
	}
	t.Logf("GGUF with metadata: %d bytes", info.Size())
}

func TestOpenSafeTensors_NotFound(t *testing.T) {
	_, err := OpenSafeTensors("/nonexistent/path/model.safetensors")
	if err == nil {
		t.Error("OpenSafeTensors should fail for nonexistent path")
	}
}

func TestReadNpy_NotFound(t *testing.T) {
	_, err := ReadNpy("/nonexistent/path/test.npy")
	if err == nil {
		t.Error("ReadNpy should fail for nonexistent path")
	}
}
