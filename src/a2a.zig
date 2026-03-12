//! A2A (Agent-to-Agent) protocol support for nullclaw.
//!
//! Implements Google's Agent-to-Agent protocol over JSON-RPC 2.0:
//!   - GET /.well-known/agent-card.json -> Agent Card discovery
//!   - POST /a2a -> JSON-RPC dispatch (message/send, message/stream, tasks/get, tasks/cancel, tasks/list)
//!
//! Task state machine: submitted -> working -> completed | failed | canceled | input_required

const std = @import("std");
const builtin = @import("builtin");
const Config = @import("config.zig").Config;
const gateway = @import("gateway.zig");
const ConversationContext = @import("agent/prompt.zig").ConversationContext;
const streaming = @import("streaming.zig");

/// Maximum number of tasks kept in the registry before eviction.
const MAX_TASKS: usize = 1000;

// ── Task State ──────────────────────────────────────────────────

pub const TaskState = enum {
    submitted,
    working,
    completed,
    failed,
    canceled,
    input_required,

    pub fn jsonName(self: TaskState) []const u8 {
        return switch (self) {
            .submitted => "submitted",
            .working => "working",
            .completed => "completed",
            .failed => "failed",
            .canceled => "canceled",
            .input_required => "input_required",
        };
    }
};

// ── Task Record ─────────────────────────────────────────────────

pub const TaskRecord = struct {
    id: []u8,
    session_key: []u8,
    state: TaskState,
    created_at: i64,
    updated_at: i64,
    user_text: []u8,
    agent_text: []u8,
};

// ── Task Registry ───────────────────────────────────────────────

pub const TaskRegistry = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    tasks: std.StringHashMapUnmanaged(*TaskRecord) = .empty,
    next_id: u64 = 1,

    pub fn init(allocator: std.mem.Allocator) TaskRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *TaskRegistry) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const task = entry.value_ptr.*;
            self.allocator.free(task.id);
            self.allocator.free(task.session_key);
            self.allocator.free(task.user_text);
            self.allocator.free(task.agent_text);
            self.allocator.destroy(task);
        }
        self.tasks.deinit(self.allocator);
    }

    pub fn createTask(self: *TaskRegistry, user_text: []const u8) !*TaskRecord {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Evict oldest completed tasks if at capacity.
        if (self.tasks.count() >= MAX_TASKS) {
            self.evictOldestCompleted();
        }

        const id_num = self.next_id;
        self.next_id += 1;

        const task_id = try std.fmt.allocPrint(self.allocator, "task-{d}", .{id_num});
        errdefer self.allocator.free(task_id);

        const session_key = try std.fmt.allocPrint(self.allocator, "a2a:{s}", .{task_id});
        errdefer self.allocator.free(session_key);

        const owned_text = try self.allocator.dupe(u8, user_text);
        errdefer self.allocator.free(owned_text);

        const empty_agent = try self.allocator.dupe(u8, "");
        errdefer self.allocator.free(empty_agent);

        const now = std.time.timestamp();

        const task = try self.allocator.create(TaskRecord);
        errdefer self.allocator.destroy(task);

        task.* = .{
            .id = task_id,
            .session_key = session_key,
            .state = .submitted,
            .created_at = now,
            .updated_at = now,
            .user_text = owned_text,
            .agent_text = empty_agent,
        };

        try self.tasks.put(self.allocator, task_id, task);

        return task;
    }

    pub fn getTask(self: *TaskRegistry, task_id: []const u8) ?*TaskRecord {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.tasks.get(task_id);
    }

    pub fn taskCount(self: *TaskRegistry) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.tasks.count();
    }

    /// List tasks with optional filtering. Returns owned slice of task pointers.
    /// Caller must free the returned slice with the provided allocator.
    pub fn listTasks(
        self: *TaskRegistry,
        allocator: std.mem.Allocator,
        filter_state: ?TaskState,
        filter_context_id: ?[]const u8,
        max_results: usize,
    ) ![]const *TaskRecord {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result: std.ArrayListUnmanaged(*TaskRecord) = .empty;
        errdefer result.deinit(allocator);

        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const task = entry.value_ptr.*;
            if (filter_state) |s| {
                if (task.state != s) continue;
            }
            if (filter_context_id) |ctx| {
                if (!std.mem.eql(u8, task.session_key, ctx)) continue;
            }
            try result.append(allocator, task);
            if (result.items.len >= max_results) break;
        }

        return result.toOwnedSlice(allocator);
    }

    /// Evict the oldest completed/failed/canceled task. Must be called with mutex held.
    fn evictOldestCompleted(self: *TaskRegistry) void {
        var oldest_key: ?[]const u8 = null;
        var oldest_time: i64 = std.math.maxInt(i64);

        var iter = self.tasks.iterator();
        while (iter.next()) |entry| {
            const task = entry.value_ptr.*;
            const is_terminal = task.state == .completed or
                task.state == .failed or
                task.state == .canceled;
            if (is_terminal and task.created_at < oldest_time) {
                oldest_time = task.created_at;
                oldest_key = entry.key_ptr.*;
            }
        }

        if (oldest_key) |key| {
            if (self.tasks.fetchRemove(key)) |kv| {
                const task = kv.value;
                self.allocator.free(task.id);
                self.allocator.free(task.session_key);
                self.allocator.free(task.user_text);
                self.allocator.free(task.agent_text);
                self.allocator.destroy(task);
            }
        }
    }
};

// ── A2A Response ────────────────────────────────────────────────

pub const A2aResponse = struct {
    status: []const u8 = "200 OK",
    body: []const u8,
    content_type: []const u8 = "application/json",
    allocated: bool = true,
};

// ── Handler: Agent Card ─────────────────────────────────────────

pub fn handleAgentCard(allocator: std.mem.Allocator, cfg: *const Config) A2aResponse {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    const w = buf.writer(allocator);
    w.writeAll("{\"name\":\"") catch return errorResponse();
    gateway.jsonEscapeInto(w, cfg.a2a.name) catch return errorResponse();
    w.writeAll("\",\"description\":\"") catch return errorResponse();
    gateway.jsonEscapeInto(w, cfg.a2a.description) catch return errorResponse();
    w.writeAll("\",\"version\":\"") catch return errorResponse();
    gateway.jsonEscapeInto(w, cfg.a2a.version) catch return errorResponse();
    // url field for backward compatibility with older A2A clients.
    w.writeAll("\",\"url\":\"") catch return errorResponse();
    gateway.jsonEscapeInto(w, cfg.a2a.url) catch return errorResponse();
    w.writeAll("/a2a") catch return errorResponse();
    // supported_interfaces per latest spec (required).
    w.writeAll("\",\"supportedInterfaces\":[{\"url\":\"") catch return errorResponse();
    gateway.jsonEscapeInto(w, cfg.a2a.url) catch return errorResponse();
    w.writeAll("/a2a\",\"protocolBinding\":\"JSONRPC\",\"protocolVersion\":\"0.2\"}]") catch return errorResponse();
    w.writeAll(",\"provider\":{\"organization\":\"") catch return errorResponse();
    gateway.jsonEscapeInto(w, cfg.a2a.name) catch return errorResponse();
    w.writeAll("\",\"url\":\"") catch return errorResponse();
    if (cfg.a2a.url.len > 0) {
        gateway.jsonEscapeInto(w, cfg.a2a.url) catch return errorResponse();
    } else {
        w.writeAll("https://github.com/nullclaw/nullclaw") catch return errorResponse();
    }
    w.writeAll("\"}") catch return errorResponse();
    w.writeAll(",\"capabilities\":{\"streaming\":true}") catch return errorResponse();
    w.writeAll(",\"defaultInputModes\":[\"text/plain\"],\"defaultOutputModes\":[\"text/plain\"]") catch return errorResponse();
    w.writeAll(",\"skills\":[{\"id\":\"chat\",\"name\":\"General Chat\",\"description\":\"General-purpose AI assistant\",\"tags\":[\"chat\",\"general\"]}]") catch return errorResponse();
    w.writeAll("}") catch return errorResponse();

    const body = buf.toOwnedSlice(allocator) catch return errorResponse();
    return .{ .body = body };
}

// ── Handler: JSON-RPC Dispatch ──────────────────────────────────

pub fn handleJsonRpc(
    allocator: std.mem.Allocator,
    body: []const u8,
    registry: *TaskRegistry,
    session_mgr: anytype,
) A2aResponse {
    const method = gateway.jsonStringField(body, "method") orelse {
        const err_body = buildJsonRpcError(allocator, "null", -32600, "Missing method") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    // Extract JSON-RPC id — may be a string or number.
    const request_id = extractJsonRpcId(body) orelse "null";

    if (std.mem.eql(u8, method, "message/send") or std.mem.eql(u8, method, "tasks/send") or
        std.mem.eql(u8, method, "message/stream") or std.mem.eql(u8, method, "tasks/sendSubscribe"))
    {
        return handleSendMessage(allocator, body, request_id, registry, session_mgr);
    } else if (std.mem.eql(u8, method, "tasks/get")) {
        return handleGetTask(allocator, body, request_id, registry);
    } else if (std.mem.eql(u8, method, "tasks/cancel")) {
        return handleCancelTask(allocator, body, request_id, registry, session_mgr);
    } else if (std.mem.eql(u8, method, "tasks/list")) {
        return handleListTasks(allocator, body, request_id, registry);
    } else {
        const err_body = buildJsonRpcError(allocator, request_id, -32601, "Method not found") catch
            return errorResponse();
        return .{ .body = err_body };
    }
}

// ── SSE Streaming ───────────────────────────────────────────────

/// Check if a JSON-RPC body contains a streaming method.
/// Used by the gateway to decide between normal and SSE response paths.
pub fn isStreamingMethod(body: []const u8) bool {
    const method = gateway.jsonStringField(body, "method") orelse return false;
    return std.mem.eql(u8, method, "message/stream") or
        std.mem.eql(u8, method, "tasks/sendSubscribe");
}

/// SSE Sink context — writes JSON-RPC SSE events to a raw TCP stream.
pub const SseStreamCtx = struct {
    stream: *std.net.Stream,
    allocator: std.mem.Allocator,
    request_id: []const u8,
    task_id: []const u8,
    filter: streaming.TagFilter = undefined,

    /// Write an SSE "data:" line with the given JSON payload.
    fn writeSseEvent(self: *SseStreamCtx, json_data: []const u8) void {
        self.stream.writeAll("data: ") catch return;
        self.stream.writeAll(json_data) catch return;
        self.stream.writeAll("\n\n") catch return;
    }

    /// Build and emit an SSE event with a working status and text delta.
    fn emitChunkEvent(self: *SseStreamCtx, text: []const u8) void {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(self.allocator);
        const w = buf.writer(self.allocator);

        w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":") catch return;
        w.writeAll(self.request_id) catch return;
        w.writeAll(",\"result\":{\"id\":\"") catch return;
        gateway.jsonEscapeInto(w, self.task_id) catch return;
        w.writeAll("\",\"status\":{\"state\":\"working\"},\"artifacts\":[{\"artifactId\":\"artifact-") catch return;
        gateway.jsonEscapeInto(w, self.task_id) catch return;
        w.writeAll("\",\"parts\":[{\"kind\":\"text\",\"text\":\"") catch return;
        gateway.jsonEscapeInto(w, text) catch return;
        w.writeAll("\"}],\"append\":true}]}}") catch return;

        const data = buf.toOwnedSlice(self.allocator) catch return;
        defer self.allocator.free(data);
        self.writeSseEvent(data);
    }

    fn sseCallback(ctx: *anyopaque, event: streaming.Event) void {
        const self: *SseStreamCtx = @ptrCast(@alignCast(ctx));
        switch (event.stage) {
            .chunk => {
                if (event.text.len > 0) self.emitChunkEvent(event.text);
            },
            .final => {}, // Final event is handled after processMessageStreaming returns.
        }
    }

    pub fn makeSink(self: *SseStreamCtx) streaming.Sink {
        const raw_sink = streaming.Sink{
            .callback = sseCallback,
            .ctx = @ptrCast(self),
        };
        self.filter = streaming.TagFilter.init(raw_sink);
        return self.filter.sink();
    }
};

/// Handle a streaming JSON-RPC request by writing SSE events directly to the TCP stream.
/// This bypasses the normal request/response cycle and writes directly.
/// The caller must NOT call writeJsonResponse after this.
pub fn handleStreamingRpc(
    allocator: std.mem.Allocator,
    body: []const u8,
    stream: *std.net.Stream,
    registry: *TaskRegistry,
    session_mgr: anytype,
) void {
    const request_id = extractJsonRpcId(body) orelse "null";

    const text = extractMessageText(body) orelse {
        writeSseError(allocator, stream, request_id, -32602, "Missing message text");
        return;
    };

    const task = registry.createTask(text) catch {
        writeSseError(allocator, stream, request_id, -32603, "Failed to create task");
        return;
    };

    // Mark as working.
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .working;
        task.updated_at = std.time.timestamp();
    }

    // Write SSE headers.
    stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n") catch return;

    // Create SSE sink context.
    var sse_ctx = SseStreamCtx{
        .stream = stream,
        .allocator = allocator,
        .request_id = request_id,
        .task_id = task.id,
    };
    const sink = sse_ctx.makeSink();

    const context: ConversationContext = .{ .channel = "a2a" };
    const response = session_mgr.processMessageStreaming(task.session_key, text, context, sink) catch {
        // Mark as failed and send error event.
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .failed;
        task.updated_at = std.time.timestamp();

        writeSseErrorEvent(allocator, &sse_ctx, -32603, "Agent processing failed");
        return;
    };

    // Update task with final response.
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .completed;
        task.updated_at = std.time.timestamp();
        registry.allocator.free(task.agent_text);
        task.agent_text = registry.allocator.dupe(u8, response) catch "";
    }

    // Send final completed event with full task JSON.
    const task_json = buildTaskJson(allocator, task) catch return;
    defer allocator.free(task_json);

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);
    const w = buf.writer(allocator);
    w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":") catch return;
    w.writeAll(request_id) catch return;
    w.writeAll(",\"result\":") catch return;
    w.writeAll(task_json) catch return;
    w.writeByte('}') catch return;

    const final_data = buf.toOwnedSlice(allocator) catch return;
    defer allocator.free(final_data);

    sse_ctx.writeSseEvent(final_data);
}

/// Write an SSE error event to the stream context.
fn writeSseErrorEvent(allocator: std.mem.Allocator, sse_ctx: *SseStreamCtx, code: i32, message: []const u8) void {
    const err_json = buildJsonRpcError(allocator, sse_ctx.request_id, code, message) catch return;
    defer allocator.free(err_json);
    sse_ctx.writeSseEvent(err_json);
}

/// Write SSE headers and a single error event for pre-streaming failures.
fn writeSseError(allocator: std.mem.Allocator, stream: *std.net.Stream, request_id: []const u8, code: i32, message: []const u8) void {
    stream.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n") catch return;
    const err_json = buildJsonRpcError(allocator, request_id, code, message) catch return;
    defer allocator.free(err_json);
    stream.writeAll("data: ") catch return;
    stream.writeAll(err_json) catch return;
    stream.writeAll("\n\n") catch return;
}

// ── Internal: Send Message ──────────────────────────────────────

fn handleSendMessage(
    allocator: std.mem.Allocator,
    body: []const u8,
    request_id: []const u8,
    registry: *TaskRegistry,
    session_mgr: anytype,
) A2aResponse {
    const text = extractMessageText(body) orelse {
        const err_body = buildJsonRpcError(allocator, request_id, -32602, "Missing message text") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    const task = registry.createTask(text) catch {
        const err_body = buildJsonRpcError(allocator, request_id, -32603, "Failed to create task") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    // Update state to working.
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .working;
        task.updated_at = std.time.timestamp();
    }

    const context: ConversationContext = .{ .channel = "a2a" };
    const response = session_mgr.processMessage(task.session_key, text, context) catch {
        // On error, mark task as failed.
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .failed;
        task.updated_at = std.time.timestamp();

        const err_body = buildJsonRpcError(allocator, request_id, -32603, "Agent processing failed") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    // Update task with response using the registry's long-lived allocator,
    // not the per-request allocator which is freed after the response is sent.
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .completed;
        task.updated_at = std.time.timestamp();
        registry.allocator.free(task.agent_text);
        task.agent_text = registry.allocator.dupe(u8, response) catch {
            const err_body = buildJsonRpcError(allocator, request_id, -32603, "Out of memory") catch
                return errorResponse();
            return .{ .body = err_body };
        };
    }

    const task_json = buildTaskJson(allocator, task) catch {
        const err_body = buildJsonRpcError(allocator, request_id, -32603, "Failed to build response") catch
            return errorResponse();
        return .{ .body = err_body };
    };
    defer allocator.free(task_json);

    const result = buildJsonRpcResult(allocator, request_id, task_json) catch
        return errorResponse();
    return .{ .body = result };
}

// ── Internal: Get Task ──────────────────────────────────────────

fn handleGetTask(
    allocator: std.mem.Allocator,
    body: []const u8,
    request_id: []const u8,
    registry: *TaskRegistry,
) A2aResponse {
    const task_id = extractParamsId(body) orelse {
        const err_body = buildJsonRpcError(allocator, request_id, -32602, "Missing task id") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    const task = registry.getTask(task_id) orelse {
        const err_body = buildJsonRpcError(allocator, request_id, -32001, "Task not found") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    const task_json = buildTaskJson(allocator, task) catch {
        const err_body = buildJsonRpcError(allocator, request_id, -32603, "Failed to build response") catch
            return errorResponse();
        return .{ .body = err_body };
    };
    defer allocator.free(task_json);

    const result = buildJsonRpcResult(allocator, request_id, task_json) catch
        return errorResponse();
    return .{ .body = result };
}

// ── Internal: Cancel Task ───────────────────────────────────────

fn handleCancelTask(
    allocator: std.mem.Allocator,
    body: []const u8,
    request_id: []const u8,
    registry: *TaskRegistry,
    session_mgr: anytype,
) A2aResponse {
    const task_id = extractParamsId(body) orelse {
        const err_body = buildJsonRpcError(allocator, request_id, -32602, "Missing task id") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    const task = registry.getTask(task_id) orelse {
        const err_body = buildJsonRpcError(allocator, request_id, -32001, "Task not found") catch
            return errorResponse();
        return .{ .body = err_body };
    };

    // Check if task is in a terminal state.
    const is_terminal = task.state == .completed or
        task.state == .failed or
        task.state == .canceled;
    if (is_terminal) {
        const err_body = buildJsonRpcError(allocator, request_id, -32002, "Task already in terminal state") catch
            return errorResponse();
        return .{ .body = err_body };
    }

    // Request interruption if working.
    if (task.state == .working) {
        var result = session_mgr.requestTurnInterrupt(task.session_key);
        result.deinit(allocator);
    }

    // Mark as canceled.
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .canceled;
        task.updated_at = std.time.timestamp();
    }

    const task_json = buildTaskJson(allocator, task) catch {
        const err_body = buildJsonRpcError(allocator, request_id, -32603, "Failed to build response") catch
            return errorResponse();
        return .{ .body = err_body };
    };
    defer allocator.free(task_json);

    const result = buildJsonRpcResult(allocator, request_id, task_json) catch
        return errorResponse();
    return .{ .body = result };
}

// ── Internal: List Tasks ────────────────────────────────────────

fn handleListTasks(
    allocator: std.mem.Allocator,
    body: []const u8,
    request_id: []const u8,
    registry: *TaskRegistry,
) A2aResponse {
    // Parse optional filters from params.
    const params_needle = "\"params\"";
    const params_section = if (std.mem.indexOf(u8, body, params_needle)) |pos|
        body[pos + params_needle.len ..]
    else
        body;

    // Optional state filter.
    const filter_state: ?TaskState = blk: {
        const state_str = gateway.jsonStringField(params_section, "state") orelse break :blk null;
        if (std.mem.eql(u8, state_str, "submitted")) break :blk .submitted;
        if (std.mem.eql(u8, state_str, "working")) break :blk .working;
        if (std.mem.eql(u8, state_str, "completed")) break :blk .completed;
        if (std.mem.eql(u8, state_str, "failed")) break :blk .failed;
        if (std.mem.eql(u8, state_str, "canceled")) break :blk .canceled;
        if (std.mem.eql(u8, state_str, "input_required")) break :blk .input_required;
        break :blk null;
    };

    // Optional context_id filter.
    const filter_context_id = gateway.jsonStringField(params_section, "contextId");

    // Page size (default 50, max 100).
    const page_size: usize = blk: {
        const val = gateway.jsonIntField(params_section, "pageSize") orelse break :blk 50;
        if (val < 1) break :blk 1;
        if (val > 100) break :blk 100;
        break :blk @intCast(val);
    };

    const tasks = registry.listTasks(allocator, filter_state, filter_context_id, page_size) catch {
        const err_body = buildJsonRpcError(allocator, request_id, -32603, "Failed to list tasks") catch
            return errorResponse();
        return .{ .body = err_body };
    };
    defer allocator.free(tasks);

    // Build result JSON: {"tasks":[...], "nextPageToken":"", "pageSize":N, "totalSize":N}
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    w.writeAll("{\"tasks\":[") catch return errorResponse();
    for (tasks, 0..) |task, i| {
        if (i > 0) w.writeByte(',') catch return errorResponse();
        const task_json = buildTaskJson(allocator, task) catch return errorResponse();
        defer allocator.free(task_json);
        w.writeAll(task_json) catch return errorResponse();
    }
    w.writeAll("],\"nextPageToken\":\"\",\"pageSize\":") catch return errorResponse();
    std.fmt.format(w, "{d}", .{page_size}) catch return errorResponse();
    w.writeAll(",\"totalSize\":") catch return errorResponse();
    std.fmt.format(w, "{d}", .{registry.taskCount()}) catch return errorResponse();
    w.writeByte('}') catch return errorResponse();

    const list_json = buf.toOwnedSlice(allocator) catch return errorResponse();
    defer allocator.free(list_json);

    const result = buildJsonRpcResult(allocator, request_id, list_json) catch
        return errorResponse();
    return .{ .body = result };
}

// ── JSON Builder Helpers ────────────────────────────────────────

/// Build a JSON-RPC result response. `request_id` is a raw JSON token (e.g. `"1"` or `1`).
fn buildJsonRpcResult(allocator: std.mem.Allocator, request_id: []const u8, result_json: []const u8) ![]u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    try w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":");
    try w.writeAll(request_id);
    try w.writeAll(",\"result\":");
    try w.writeAll(result_json);
    try w.writeByte('}');

    return buf.toOwnedSlice(allocator);
}

/// Build a JSON-RPC error response. `request_id` is a raw JSON token (e.g. `"1"` or `1`).
fn buildJsonRpcError(allocator: std.mem.Allocator, request_id: []const u8, code: i32, message: []const u8) ![]u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    try w.writeAll("{\"jsonrpc\":\"2.0\",\"id\":");
    try w.writeAll(request_id);
    try w.writeAll(",\"error\":{\"code\":");
    try std.fmt.format(w, "{d}", .{code});
    try w.writeAll(",\"message\":\"");
    try gateway.jsonEscapeInto(w, message);
    try w.writeAll("\"}}");

    return buf.toOwnedSlice(allocator);
}

fn buildTaskJson(allocator: std.mem.Allocator, task: *const TaskRecord) ![]u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);
    const w = buf.writer(allocator);

    // Format timestamp as ISO 8601 from unix epoch seconds.
    var ts_buf: [32]u8 = undefined;
    const timestamp = formatTimestamp(&ts_buf, task.updated_at);

    try w.writeAll("{\"id\":\"");
    try gateway.jsonEscapeInto(w, task.id);
    try w.writeAll("\",\"contextId\":\"");
    try gateway.jsonEscapeInto(w, task.session_key);
    try w.writeAll("\",\"status\":{\"state\":\"");
    try w.writeAll(task.state.jsonName());
    try w.writeAll("\",\"timestamp\":\"");
    try w.writeAll(timestamp);
    try w.writeAll("\"}");

    // Include artifacts and history when agent_text is non-empty.
    if (task.agent_text.len > 0) {
        try w.writeAll(",\"artifacts\":[{\"artifactId\":\"artifact-");
        try gateway.jsonEscapeInto(w, task.id);
        try w.writeAll("\",\"parts\":[{\"kind\":\"text\",\"text\":\"");
        try gateway.jsonEscapeInto(w, task.agent_text);
        try w.writeAll("\"}]}]");

        try w.writeAll(",\"history\":[{\"role\":\"user\",\"messageId\":\"msg-user-");
        try gateway.jsonEscapeInto(w, task.id);
        try w.writeAll("\",\"parts\":[{\"kind\":\"text\",\"text\":\"");
        try gateway.jsonEscapeInto(w, task.user_text);
        try w.writeAll("\"}]},{\"role\":\"agent\",\"messageId\":\"msg-agent-");
        try gateway.jsonEscapeInto(w, task.id);
        try w.writeAll("\",\"parts\":[{\"kind\":\"text\",\"text\":\"");
        try gateway.jsonEscapeInto(w, task.agent_text);
        try w.writeAll("\"}]}]");
    }

    try w.writeByte('}');

    return buf.toOwnedSlice(allocator);
}

/// Format a unix timestamp (seconds since epoch) as ISO 8601 UTC.
fn formatTimestamp(buf: *[32]u8, epoch_secs: i64) []const u8 {
    const epoch_day = @divFloor(epoch_secs, 86400);
    const day_secs: u32 = @intCast(@mod(epoch_secs, 86400));
    const hours = day_secs / 3600;
    const mins = (day_secs % 3600) / 60;
    const secs = day_secs % 60;

    // Civil date from epoch day (algorithm from Howard Hinnant).
    const z: i64 = epoch_day + 719468;
    const era: i64 = @divFloor(if (z >= 0) z else z - 146096, 146097);
    const doe: u32 = @intCast(z - era * 146097);
    const yoe: u32 = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const y: i64 = @as(i64, @intCast(yoe)) + era * 400;
    const doy: u32 = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const mp: u32 = (5 * doy + 2) / 153;
    const d: u32 = doy - (153 * mp + 2) / 5 + 1;
    const m: u32 = if (mp < 10) mp + 3 else mp - 9;
    const year: i64 = if (m <= 2) y + 1 else y;

    const result = std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        @as(u32, @intCast(year)), m, d, hours, mins, secs,
    }) catch "1970-01-01T00:00:00Z";
    return result;
}

// ── Text Extraction Helpers ─────────────────────────────────────

/// Extract the user's message text from A2A params.message.parts[0].text.
/// Scans for "text" key after "parts", skipping occurrences that appear as
/// values (e.g. "type":"text") by retrying when jsonStringField returns null.
fn extractMessageText(body: []const u8) ?[]const u8 {
    const parts_needle = "\"parts\"";
    const parts_pos = std.mem.indexOf(u8, body, parts_needle) orelse return null;
    var remaining = body[parts_pos + parts_needle.len ..];

    const key_needle = "\"text\"";
    // Iterate through all occurrences of "text" to find one that is a JSON key.
    while (std.mem.indexOf(u8, remaining, key_needle)) |pos| {
        const after_key = remaining[pos + key_needle.len ..];
        // Skip whitespace, then check for colon (key indicator).
        var i: usize = 0;
        while (i < after_key.len and (after_key[i] == ' ' or after_key[i] == '\t' or
            after_key[i] == '\n' or after_key[i] == '\r')) : (i += 1)
        {}
        if (i < after_key.len and after_key[i] == ':') {
            // This is a key — use jsonStringField from this position.
            return gateway.jsonStringField(remaining[pos..], "text");
        }
        // Not a key (it's a value like "type":"text"), skip past and continue.
        remaining = remaining[pos + key_needle.len ..];
    }
    return null;
}

/// Extract the JSON-RPC "id" field as a raw JSON token (string including quotes, or number).
/// Handles both `"id": "abc"` and `"id": 123`.
fn extractJsonRpcId(body: []const u8) ?[]const u8 {
    const needle = "\"id\"";
    // Find top-level "id" — it appears before "params" in a well-formed request.
    const id_pos = std.mem.indexOf(u8, body, needle) orelse return null;
    const after_key = body[id_pos + needle.len ..];

    // Skip whitespace and colon.
    var i: usize = 0;
    while (i < after_key.len and (after_key[i] == ' ' or after_key[i] == ':' or
        after_key[i] == '\t' or after_key[i] == '\n' or after_key[i] == '\r')) : (i += 1)
    {}
    if (i >= after_key.len) return null;

    if (after_key[i] == '"') {
        // String value — return including quotes for raw JSON embedding.
        const start = i;
        i += 1;
        while (i < after_key.len) : (i += 1) {
            if (after_key[i] == '\\' and i + 1 < after_key.len) {
                i += 1;
                continue;
            }
            if (after_key[i] == '"') {
                return after_key[start .. i + 1]; // includes both quotes
            }
        }
        return null;
    } else if (after_key[i] == '-' or (after_key[i] >= '0' and after_key[i] <= '9')) {
        // Numeric value — scan digits.
        const start = i;
        while (i < after_key.len and ((after_key[i] >= '0' and after_key[i] <= '9') or
            after_key[i] == '-' or after_key[i] == '.' or
            after_key[i] == 'e' or after_key[i] == 'E' or after_key[i] == '+')) : (i += 1)
        {}
        if (i > start) return after_key[start..i];
        return null;
    }
    return null;
}

/// Extract task ID from params.id in the JSON-RPC body.
fn extractParamsId(body: []const u8) ?[]const u8 {
    const params_needle = "\"params\"";
    const params_pos = std.mem.indexOf(u8, body, params_needle) orelse
        return gateway.jsonStringField(body, "id");
    const after_params = body[params_pos + params_needle.len ..];
    return gateway.jsonStringField(after_params, "id");
}

// ── Fallback Error Response ─────────────────────────────────────

fn errorResponse() A2aResponse {
    return .{
        .status = "500 Internal Server Error",
        .body = "{\"error\":\"internal server error\"}",
        .allocated = false,
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

const testing = std.testing;

const MockSessionManager = struct {
    response: []const u8 = "mock response",

    pub fn processMessage(self: *MockSessionManager, _: []const u8, _: []const u8, _: anytype) ![]const u8 {
        return self.response;
    }

    pub fn processMessageStreaming(self: *MockSessionManager, session_key: []const u8, content: []const u8, conversation_context: anytype, sink: ?streaming.Sink) ![]const u8 {
        // Emit chunks if a sink is provided.
        if (sink) |s| {
            s.emitChunk(self.response);
            s.emitFinal();
        }
        return self.processMessage(session_key, content, conversation_context);
    }

    pub fn requestTurnInterrupt(_: *MockSessionManager, _: []const u8) struct {
        requested: bool,
        active_tool: ?[]u8,

        pub fn deinit(s: *@This(), _: anytype) void {
            _ = s;
        }
    } {
        return .{ .requested = false, .active_tool = null };
    }
};

fn testConfig() Config {
    return .{
        .workspace_dir = "/tmp/a2a_test",
        .config_path = "/tmp/a2a_test/config.json",
        .default_model = "test/mock-model",
        .allocator = testing.allocator,
        .a2a = .{
            .enabled = true,
            .name = "TestAgent",
            .description = "A test agent",
            .url = "http://localhost:3000",
            .version = "1.0.0",
        },
    };
}

test "TaskState jsonName returns correct strings" {
    try testing.expectEqualStrings("submitted", TaskState.submitted.jsonName());
    try testing.expectEqualStrings("working", TaskState.working.jsonName());
    try testing.expectEqualStrings("completed", TaskState.completed.jsonName());
    try testing.expectEqualStrings("failed", TaskState.failed.jsonName());
    try testing.expectEqualStrings("canceled", TaskState.canceled.jsonName());
    try testing.expectEqualStrings("input_required", TaskState.input_required.jsonName());
}

test "TaskRegistry createTask and getTask" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const task = try registry.createTask("hello world");
    try testing.expectEqualStrings("task-1", task.id);
    try testing.expectEqualStrings("a2a:task-1", task.session_key);
    try testing.expectEqualStrings("hello world", task.user_text);
    try testing.expect(task.state == .submitted);
    try testing.expectEqual(@as(usize, 0), task.agent_text.len);

    const found = registry.getTask("task-1");
    try testing.expect(found != null);
    try testing.expect(found.? == task);

    const not_found = registry.getTask("task-999");
    try testing.expect(not_found == null);

    try testing.expectEqual(@as(usize, 1), registry.taskCount());
}

test "TaskRegistry evicts oldest completed tasks" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    // Fill registry to MAX_TASKS with completed tasks.
    var i: usize = 0;
    while (i < MAX_TASKS) : (i += 1) {
        const task = try registry.createTask("filler");
        registry.mutex.lock();
        task.state = .completed;
        task.created_at = @as(i64, @intCast(i)); // ascending created_at
        registry.mutex.unlock();
    }
    try testing.expectEqual(MAX_TASKS, registry.taskCount());

    // Creating one more should evict the oldest (task-1 with created_at=0).
    _ = try registry.createTask("new task");
    try testing.expectEqual(MAX_TASKS, registry.taskCount());

    // task-1 should be evicted (it had the oldest created_at).
    try testing.expect(registry.getTask("task-1") == null);

    // The newest task should exist.
    const newest_id = try std.fmt.allocPrint(testing.allocator, "task-{d}", .{MAX_TASKS + 1});
    defer testing.allocator.free(newest_id);
    try testing.expect(registry.getTask(newest_id) != null);
}

test "handleAgentCard returns valid JSON" {
    const cfg = testConfig();
    const resp = handleAgentCard(testing.allocator, &cfg);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expectEqualStrings("application/json", resp.content_type);

    // Verify key fields are present.
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"name\":\"TestAgent\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"description\":\"A test agent\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"version\":\"1.0.0\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"url\":\"http://localhost:3000/a2a\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"supportedInterfaces\":[") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"protocolBinding\":\"JSONRPC\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"provider\":{\"organization\":\"TestAgent\",\"url\":\"http://localhost:3000\"}") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"streaming\":true") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"defaultInputModes\":[\"text/plain\"]") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"skills\":[") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"tags\":[\"chat\",\"general\"]") != null);
}

test "handleJsonRpc dispatches tasks/send" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-1","method":"tasks/send","params":{"message":{"role":"user","parts":[{"type":"text","text":"Hello agent"}]}}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"result\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"id\":\"req-1\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"completed\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "mock response") != null);

    // Verify task was created in registry.
    try testing.expectEqual(@as(usize, 1), registry.taskCount());
}

test "handleJsonRpc dispatches message/send" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":42,"method":"message/send","params":{"message":{"role":"user","parts":[{"type":"text","text":"Hello via message/send"}]}}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"id\":42") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"completed\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "mock response") != null);
}

test "handleJsonRpc dispatches tasks/get" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    // Create a task first.
    const task = try registry.createTask("test input");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .completed;
        testing.allocator.free(task.agent_text);
        task.agent_text = try testing.allocator.dupe(u8, "test output");
    }

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-2","method":"tasks/get","params":{"id":"task-1"}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"result\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"id\":\"req-2\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"completed\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "test output") != null);
}

test "handleJsonRpc returns error for unknown method" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-3","method":"unknown/method","params":{}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expect(std.mem.indexOf(u8, resp.body, "\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "-32601") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "Method not found") != null);
}

test "handleJsonRpc returns error for missing method" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-4","params":{}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expect(std.mem.indexOf(u8, resp.body, "\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "-32600") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "Missing method") != null);
}

test "buildTaskJson escapes special characters" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const task = try registry.createTask("hello \"world\"");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .completed;
        testing.allocator.free(task.agent_text);
        task.agent_text = try testing.allocator.dupe(u8, "line1\nline2\ttab");
    }

    const json = try buildTaskJson(testing.allocator, task);
    defer testing.allocator.free(json);

    // The escaped quotes should appear as \"
    try testing.expect(std.mem.indexOf(u8, json, "hello \\\"world\\\"") != null);
    // Newline and tab should be escaped.
    try testing.expect(std.mem.indexOf(u8, json, "line1\\nline2\\ttab") != null);
}

test "extractMessageText finds text in parts" {
    const body =
        \\{"jsonrpc":"2.0","id":"1","method":"tasks/send","params":{"message":{"role":"user","parts":[{"type":"text","text":"Hello there"}]}}}
    ;
    const text = extractMessageText(body);
    try testing.expect(text != null);
    try testing.expectEqualStrings("Hello there", text.?);
}

test "extractMessageText returns null for missing parts" {
    const body =
        \\{"jsonrpc":"2.0","id":"1","method":"tasks/send","params":{"message":{"role":"user"}}}
    ;
    const text = extractMessageText(body);
    try testing.expect(text == null);
}

test "handleJsonRpc dispatches tasks/cancel" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    // Create a working task.
    const task = try registry.createTask("cancel me");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .working;
    }

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-5","method":"tasks/cancel","params":{"id":"task-1"}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"canceled\"") != null);
    try testing.expect(task.state == .canceled);
}

test "handleJsonRpc cancel returns error for completed task" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const task = try registry.createTask("done");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .completed;
    }

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-6","method":"tasks/cancel","params":{"id":"task-1"}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expect(std.mem.indexOf(u8, resp.body, "\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "-32002") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "terminal") != null);
}

test "handleJsonRpc tasks/get returns error for nonexistent task" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-7","method":"tasks/get","params":{"id":"task-nonexistent"}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expect(std.mem.indexOf(u8, resp.body, "\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "-32001") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "Task not found") != null);
}

test "handleJsonRpc dispatches tasks/sendSubscribe same as tasks/send" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-8","method":"tasks/sendSubscribe","params":{"message":{"role":"user","parts":[{"type":"text","text":"Subscribe test"}]}}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"result\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"completed\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "mock response") != null);
}

test "buildTaskJson omits artifacts and history when agent_text is empty" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const task = try registry.createTask("test input");

    const json = try buildTaskJson(testing.allocator, task);
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "\"artifacts\"") == null);
    try testing.expect(std.mem.indexOf(u8, json, "\"history\"") == null);
    try testing.expect(std.mem.indexOf(u8, json, "\"status\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"contextId\"") != null);
}

test "buildTaskJson includes contextId artifactId and messageId" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const task = try registry.createTask("test");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        task.state = .completed;
        testing.allocator.free(task.agent_text);
        task.agent_text = try testing.allocator.dupe(u8, "reply");
    }

    const json = try buildTaskJson(testing.allocator, task);
    defer testing.allocator.free(json);

    try testing.expect(std.mem.indexOf(u8, json, "\"contextId\":\"a2a:task-1\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"artifactId\":\"artifact-task-1\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"messageId\":\"msg-user-task-1\"") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"messageId\":\"msg-agent-task-1\"") != null);
}

test "extractJsonRpcId handles string id" {
    const body =
        \\{"jsonrpc":"2.0","id":"req-1","method":"tasks/send"}
    ;
    const id = extractJsonRpcId(body);
    try testing.expect(id != null);
    try testing.expectEqualStrings("\"req-1\"", id.?);
}

test "extractJsonRpcId handles numeric id" {
    const body =
        \\{"jsonrpc":"2.0","id":42,"method":"tasks/send"}
    ;
    const id = extractJsonRpcId(body);
    try testing.expect(id != null);
    try testing.expectEqualStrings("42", id.?);
}

test "extractJsonRpcId returns null when missing" {
    const body =
        \\{"jsonrpc":"2.0","method":"tasks/send"}
    ;
    const id = extractJsonRpcId(body);
    try testing.expect(id == null);
}

test "extractParamsId finds id in params" {
    const body =
        \\{"jsonrpc":"2.0","id":"req-1","method":"tasks/get","params":{"id":"task-42"}}
    ;
    const id = extractParamsId(body);
    try testing.expect(id != null);
    try testing.expectEqualStrings("task-42", id.?);
}

test "multiple tasks get unique IDs" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const t1 = try registry.createTask("first");
    const t2 = try registry.createTask("second");
    const t3 = try registry.createTask("third");

    try testing.expectEqualStrings("task-1", t1.id);
    try testing.expectEqualStrings("task-2", t2.id);
    try testing.expectEqualStrings("task-3", t3.id);
    try testing.expectEqual(@as(usize, 3), registry.taskCount());
}

test "handleJsonRpc dispatches tasks/list" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    // Create some tasks with different states.
    const t1 = try registry.createTask("first");
    const t2 = try registry.createTask("second");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        t1.state = .completed;
        testing.allocator.free(t1.agent_text);
        t1.agent_text = try testing.allocator.dupe(u8, "reply1");
        t2.state = .working;
    }

    var mock = MockSessionManager{};
    const body =
        \\{"jsonrpc":"2.0","id":"req-list","method":"tasks/list","params":{}}
    ;
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"result\"") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"tasks\":[") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"totalSize\":2") != null);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"pageSize\":50") != null);
}

test "handleListTasks filters by state" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    const t1 = try registry.createTask("first");
    _ = try registry.createTask("second");
    {
        registry.mutex.lock();
        defer registry.mutex.unlock();
        t1.state = .completed;
        testing.allocator.free(t1.agent_text);
        t1.agent_text = try testing.allocator.dupe(u8, "done");
    }

    const body =
        \\{"jsonrpc":"2.0","id":"req-f","method":"tasks/list","params":{"state":"completed"}}
    ;
    var mock = MockSessionManager{};
    const resp = handleJsonRpc(testing.allocator, body, &registry, &mock);
    defer if (resp.allocated) testing.allocator.free(resp.body);

    try testing.expectEqualStrings("200 OK", resp.status);
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"completed\"") != null);
    // Only the completed task should appear.
    try testing.expect(std.mem.indexOf(u8, resp.body, "\"state\":\"submitted\"") == null);
}

test "listTasks returns empty slice when no tasks match" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    _ = try registry.createTask("hello");

    const tasks = try registry.listTasks(testing.allocator, .canceled, null, 50);
    defer testing.allocator.free(tasks);

    try testing.expectEqual(@as(usize, 0), tasks.len);
}

test "isStreamingMethod detects message/stream" {
    const body =
        \\{"jsonrpc":"2.0","id":1,"method":"message/stream","params":{"message":{"role":"user","parts":[{"type":"text","text":"hi"}]}}}
    ;
    try testing.expect(isStreamingMethod(body));
}

test "isStreamingMethod detects tasks/sendSubscribe" {
    const body =
        \\{"jsonrpc":"2.0","id":"1","method":"tasks/sendSubscribe","params":{}}
    ;
    try testing.expect(isStreamingMethod(body));
}

test "isStreamingMethod returns false for tasks/send" {
    const body =
        \\{"jsonrpc":"2.0","id":"1","method":"tasks/send","params":{}}
    ;
    try testing.expect(!isStreamingMethod(body));
}

test "isStreamingMethod returns false for message/send" {
    const body =
        \\{"jsonrpc":"2.0","id":"1","method":"message/send","params":{}}
    ;
    try testing.expect(!isStreamingMethod(body));
}

test "listTasks respects max_results" {
    var registry = TaskRegistry.init(testing.allocator);
    defer registry.deinit();

    _ = try registry.createTask("a");
    _ = try registry.createTask("b");
    _ = try registry.createTask("c");

    const tasks = try registry.listTasks(testing.allocator, null, null, 2);
    defer testing.allocator.free(tasks);

    try testing.expectEqual(@as(usize, 2), tasks.len);
}
