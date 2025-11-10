package main

// LLMClient is an interface for different LLM providers
type LLMClient interface {
	GetEmbedding(text string) ([]float64, error)
	Chat(messages []Message) (string, error)
}

// ensure all clients implement the interface
var _ LLMClient = (*OpenAIClient)(nil)
var _ LLMClient = (*HybridClient)(nil)
var _ LLMClient = (*VoyageClaudeClient)(nil)

// HybridClient uses OpenAI for embeddings and Claude for chat
type HybridClient struct {
	OpenAI *OpenAIClient
	Claude *AnthropicClient
}

// NewHybridClient creates a client that uses OpenAI for embeddings and Claude for chat
func NewHybridClient(openaiKey, claudeKey string) *HybridClient {
	return &HybridClient{
		OpenAI: NewOpenAIClient(openaiKey),
		Claude: NewAnthropicClient(claudeKey),
	}
}

// GetEmbedding uses OpenAI for embeddings
func (h *HybridClient) GetEmbedding(text string) ([]float64, error) {
	return h.OpenAI.GetEmbedding(text)
}

// Chat uses Claude for chat completions
func (h *HybridClient) Chat(messages []Message) (string, error) {
	return h.Claude.Chat(messages)
}
