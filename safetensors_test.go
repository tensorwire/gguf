package gguf

import (
	"fmt"
	"os"
	"testing"
)

func TestSafeTensorsOpen(t *testing.T) {
	// Try to open ReluLLaMA-7B if available
	paths := []string{
		"/mnt/data/tensorwire/ReluLLaMA-7B",
		os.Getenv("HOME") + "/models/ReluLLaMA-7B",
	}

	var st *SafeTensors
	var err error
	for _, p := range paths {
		st, err = OpenSafeTensors(p)
		if err == nil {
			fmt.Printf("Opened: %s (%d tensors)\n", p, len(st.TensorNames))
			break
		}
	}

	if st == nil {
		t.Skip("no safetensors model found")
	}

	// Print first 10 tensor names + shapes
	fmt.Println("\nFirst 20 tensors:")
	for i, name := range st.TensorNames {
		if i >= 20 {
			break
		}
		ti, _ := st.GetInfo(name)
		fmt.Printf("  %-60s  dtype=%-4s  shape=%v  bytes=%d\n",
			name, ti.Dtype, ti.Shape, ti.ByteSize())
	}

	// Read one tensor and verify
	testName := "model.layers.0.mlp.gate_proj.weight"
	data, ti, err := st.ReadTensorFloat32(testName)
	if err != nil {
		t.Fatalf("ReadTensorFloat32(%s): %v", testName, err)
	}
	fmt.Printf("\nRead %s: %d elements, shape=%v, first 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
		testName, len(data), ti.Shape, data[0], data[1], data[2], data[3], data[4])

	// Verify element count matches shape
	expected := ti.NumElements()
	if len(data) != expected {
		t.Errorf("element count mismatch: got %d, expected %d", len(data), expected)
	}

	// Verify non-zero (weights shouldn't all be zero)
	nonZero := 0
	for _, v := range data {
		if v != 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Errorf("all zeros — something is wrong")
	}
	fmt.Printf("  Non-zero: %d/%d (%.1f%%)\n", nonZero, len(data), float64(nonZero)/float64(len(data))*100)

	// List all layer 0 tensors
	l0 := st.ListTensors("model.layers.0.")
	fmt.Printf("\nLayer 0 tensors (%d):\n", len(l0))
	for _, name := range l0 {
		ti, _ := st.GetInfo(name)
		fmt.Printf("  %-55s  %s %v\n", name, ti.Dtype, ti.Shape)
	}
}
