package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// AnthropicClient handles Anthropic API requests
type AnthropicClient struct {
	APIKey string
	Client *http.Client
}

// NewAnthropicClient creates a new Anthropic client
func NewAnthropicClient(apiKey string) *AnthropicClient {
	return &AnthropicClient{
		APIKey: apiKey,
		Client: &http.Client{},
	}
}

// GetEmbedding gets an embedding using Voyage AI (Anthropic's recommended provider)
// Note: Anthropic doesn't provide embeddings directly, so we still need OpenAI or Voyage
// For simplicity, we'll use a wrapper that falls back to OpenAI embeddings
func (c *AnthropicClient) GetEmbedding(text string) ([]float64, error) {
	// anthropic doesn't provide embeddings, so we need to use openai for this part
	// you could also use voyage ai or other embedding providers
	return nil, fmt.Errorf("embeddings not supported directly by anthropic - use openai for embeddings")
}

// ChatRequest represents an Anthropic messages API request
type AnthropicChatRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Messages  []AnthropicMessage `json:"messages"`
	System    string             `json:"system,omitempty"`
}

// AnthropicMessage represents a message in the chat
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnthropicChatResponse represents the response
type AnthropicChatResponse struct {
	Content []struct {
		Text string `json:"text"`
		Type string `json:"type"`
	} `json:"content"`
}

// Chat sends a chat completion request to Claude
func (c *AnthropicClient) Chat(messages []Message) (string, error) {
	// separate system message from user messages
	var systemPrompt string
	var userMessages []AnthropicMessage

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
		} else {
			userMessages = append(userMessages, AnthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	reqBody := AnthropicChatRequest{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 4096,
		Messages:  userMessages,
		System:    systemPrompt,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.Client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("anthropic api error: %s - %s", resp.Status, string(bodyBytes))
	}

	var chatResp AnthropicChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", err
	}

	if len(chatResp.Content) == 0 {
		return "", fmt.Errorf("no response from claude")
	}

	return chatResp.Content[0].Text, nil
}
