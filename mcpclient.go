package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
)

// mcpclient handles communication with a running MCP server

type mcpRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type mcpResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *mcpError       `json:"error,omitempty"`
}

type mcpError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type toolCallParams struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

type toolCallResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

// queryViaMCP sends a query to the running MCP server
func queryViaMCP(query string, topK int, synthesize bool) (string, error) {
	// find the lr binary path
	lrPath, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("failed to find lr binary: %w", err)
	}

	// resolve symlinks
	lrPath, err = os.Readlink(lrPath)
	if err != nil {
		// not a symlink, use as-is
		lrPath, _ = os.Executable()
	}

	// start the mcp server as a subprocess
	cmd := exec.Command(lrPath, "mcp", "--no-preload")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	// discard stderr to avoid interfering with JSON-RPC
	cmd.Stderr = nil

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start MCP server: %w", err)
	}
	defer cmd.Process.Kill()

	// send initialize request
	initReq := mcpRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "lr-cli",
				"version": "1.0.0",
			},
		},
	}

	if err := json.NewEncoder(stdin).Encode(initReq); err != nil {
		return "", fmt.Errorf("failed to send initialize: %w", err)
	}

	// read initialize response
	reader := bufio.NewReader(stdout)
	var initResp mcpResponse
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return "", fmt.Errorf("failed to read initialize response: %w", err)
	}
	if err := json.Unmarshal(line, &initResp); err != nil {
		return "", fmt.Errorf("failed to parse initialize response: %w", err)
	}

	if initResp.Error != nil {
		return "", fmt.Errorf("initialize error: %s", initResp.Error.Message)
	}

	// send initialized notification (notifications don't get responses)
	initializedNotif := map[string]interface{}{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
	}
	if err := json.NewEncoder(stdin).Encode(initializedNotif); err != nil {
		return "", fmt.Errorf("failed to send initialized notification: %w", err)
	}

	// send tool call
	toolReq := mcpRequest{
		JSONRPC: "2.0",
		ID:      2,
		Method:  "tools/call",
		Params: toolCallParams{
			Name: "query_repositories",
			Arguments: map[string]interface{}{
				"query":      query,
				"top_k":      float64(topK),
				"synthesize": synthesize,
			},
		},
	}

	if err := json.NewEncoder(stdin).Encode(toolReq); err != nil {
		return "", fmt.Errorf("failed to send tool call: %w", err)
	}

	// read tool response
	line, err = reader.ReadBytes('\n')
	if err != nil && err != io.EOF {
		return "", fmt.Errorf("failed to read tool response: %w", err)
	}

	var toolResp mcpResponse
	if err := json.Unmarshal(line, &toolResp); err != nil {
		// debug: show what we received
		return "", fmt.Errorf("failed to parse tool response: %w (received: %s)", err, string(line))
	}

	if toolResp.Error != nil {
		return "", fmt.Errorf("tool call error: %s", toolResp.Error.Message)
	}

	// parse result
	var result toolCallResult
	if err := json.Unmarshal(toolResp.Result, &result); err != nil {
		return "", fmt.Errorf("failed to parse tool result: %w", err)
	}

	if len(result.Content) == 0 || result.Content[0].Type != "text" {
		return "", fmt.Errorf("unexpected result format")
	}

	return result.Content[0].Text, nil
}
