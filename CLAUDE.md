# CLAUDE.md — GGUF

## What This Is

Go-native model serialization library. Read and write GGUF (llama.cpp/Ollama), SafeTensors (HuggingFace), and NumPy (.npy) formats. Zero Python dependencies. Zero external dependencies — stdlib only.

## Build

```bash
go build ./...
go test -v ./...
```

## Architecture

- `gguf.go` — GGUF file writer with F32, Q8_0 (~3.5x compression), and Q4_0 (~6.2x compression) quantization
- `safetensors.go` — SafeTensors reader/writer with F16→F32 and BF16→F32 dtype conversion, sharded model support, INT8 quantization
- `numpy.go` — NumPy .npy reader/writer

## Key Functions

```go
// GGUF
w := gguf.NewGGUFWriter()
w.AddTensorQ8_0("weight", data, 4096, 4096)
w.Write("model.gguf")

// SafeTensors
st, _ := gguf.OpenSafeTensors("model.safetensors")
data, info, _ := st.ReadTensorFloat32("model.layers.0.weight")
gguf.SaveSafeTensors("out.safetensors", tensors)

// NumPy
arr, _ := gguf.ReadNpy("weights.npy")
gguf.WriteNpy("out.npy", data, 2, 3)
```

## Related Packages

- `github.com/tensorwire/mongoose` — GPU compute engine
- `github.com/tensorwire/tokenizer` — BPE tokenizer
