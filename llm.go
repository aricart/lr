package main

import (
	"fmt"
	"strings"
	"time"
)

// LLMClient is an interface for different LLM providers
type LLMClient interface {
	GetEmbedding(text string) ([]float64, error)
	Chat(messages []Message) (string, error)
}

// getEmbeddingWithRetry retries transient API errors with exponential backoff.
func getEmbeddingWithRetry(client LLMClient, text string, maxAttempts int) ([]float64, error) {
	return doEmbeddingWithRetry(client, text, maxAttempts, time.Sleep)
}

func doEmbeddingWithRetry(client LLMClient, text string, maxAttempts int, sleep func(time.Duration)) ([]float64, error) {
	var lastErr error
	for attempt := range maxAttempts {
		embedding, err := client.GetEmbedding(text)
		if err == nil {
			return embedding, nil
		}
		lastErr = err
		if !isRetryableError(err) {
			return nil, err
		}
		if attempt < maxAttempts-1 {
			delay := time.Duration(1<<attempt) * time.Second
			fmt.Printf("\nretryable error (attempt %d/%d), waiting %s: %v\n", attempt+1, maxAttempts, delay, err)
			sleep(delay)
		}
	}
	return nil, fmt.Errorf("failed after %d attempts: %w", maxAttempts, lastErr)
}

// isRetryableError returns true for transient server errors and rate limits.
func isRetryableError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "500") ||
		strings.Contains(msg, "502") ||
		strings.Contains(msg, "503") ||
		strings.Contains(msg, "504") ||
		strings.Contains(msg, "429") ||
		strings.Contains(msg, "rate limit")
}

// ensure all clients implement the interface
var _ LLMClient = (*OpenAIClient)(nil)
var _ LLMClient = (*HybridClient)(nil)
var _ LLMClient = (*VoyageClaudeClient)(nil)
var _ LLMClient = (*OllamaClaudeClient)(nil)

// HybridClient uses OpenAI for embeddings and Claude for chat
type HybridClient struct {
	OpenAI *OpenAIClient
	Claude *AnthropicClient
}

// NewHybridClient creates a client that uses OpenAI for embeddings and Claude for chat
func NewHybridClient(openaiKey, claudeKey, embeddingModel, chatModel string) *HybridClient {
	return &HybridClient{
		OpenAI: NewOpenAIClient(openaiKey, "", embeddingModel), // empty chat model since we use Claude for chat
		Claude: NewAnthropicClient(claudeKey, chatModel),
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
