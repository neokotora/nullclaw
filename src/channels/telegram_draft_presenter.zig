const std = @import("std");

pub const DRAFT_FLUSH_MIN_DELTA_BYTES: usize = 16;
pub const DRAFT_FLUSH_MIN_INTERVAL_MS: i64 = 200;

const DRAFT_TRIM_BYTES = " \t\r\n";

pub const DraftState = struct {
    draft_id: u64,
    buffer: std.ArrayListUnmanaged(u8) = .empty,
    last_flush_len: usize = 0,
    last_flush_time: i64 = 0,

    pub fn deinit(self: *DraftState, allocator: std.mem.Allocator) void {
        self.buffer.deinit(allocator);
    }
};

pub const DraftFlush = struct {
    draft_id: u64,
    text: []u8,

    pub fn deinit(self: *DraftFlush, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

fn trimmedDraftText(text: []const u8) []const u8 {
    return std.mem.trim(u8, text, DRAFT_TRIM_BYTES);
}

pub fn hasVisibleDraftText(text: []const u8) bool {
    return trimmedDraftText(text).len != 0;
}

fn bytesSinceLastFlush(state: *const DraftState) usize {
    return state.buffer.items.len - state.last_flush_len;
}

fn millisSinceLastFlush(state: *const DraftState, now_ms: i64) i64 {
    return now_ms - state.last_flush_time;
}

fn flushDeltaReached(state: *const DraftState) bool {
    return bytesSinceLastFlush(state) >= DRAFT_FLUSH_MIN_DELTA_BYTES;
}

fn flushIntervalElapsed(state: *const DraftState, now_ms: i64) bool {
    return millisSinceLastFlush(state, now_ms) >= DRAFT_FLUSH_MIN_INTERVAL_MS;
}

fn shouldFlushDraft(state: *const DraftState, now_ms: i64) bool {
    return flushDeltaReached(state) or flushIntervalElapsed(state, now_ms);
}

fn snapshotDraftText(allocator: std.mem.Allocator, state: *const DraftState) ![]u8 {
    return allocator.dupe(u8, state.buffer.items);
}

fn markDraftFlushed(state: *DraftState, now_ms: i64) void {
    state.last_flush_len = state.buffer.items.len;
    state.last_flush_time = now_ms;
}

pub fn appendDraftChunk(
    allocator: std.mem.Allocator,
    state: *DraftState,
    chunk: []const u8,
    now_ms: i64,
) !?DraftFlush {
    if (chunk.len == 0) return null;

    try state.buffer.appendSlice(allocator, chunk);
    if (!shouldFlushDraft(state, now_ms)) return null;
    if (!hasVisibleDraftText(state.buffer.items)) return null;

    const text = try snapshotDraftText(allocator, state);
    markDraftFlushed(state, now_ms);
    return .{
        .draft_id = state.draft_id,
        .text = text,
    };
}

pub fn clearDraftForTarget(
    allocator: std.mem.Allocator,
    draft_buffers: *std.StringHashMapUnmanaged(DraftState),
    target: []const u8,
) void {
    if (draft_buffers.fetchRemove(target)) |entry| {
        allocator.free(entry.key);
        var draft = entry.value;
        draft.deinit(allocator);
    }
}

pub fn deinitDraftBuffers(
    allocator: std.mem.Allocator,
    draft_buffers: *std.StringHashMapUnmanaged(DraftState),
) void {
    var it = draft_buffers.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit(allocator);
    }
    draft_buffers.deinit(allocator);
    draft_buffers.* = .empty;
}

test "appendDraftChunk ignores empty chunks" {
    var draft: DraftState = .{ .draft_id = 1 };
    defer draft.deinit(std.testing.allocator);

    try std.testing.expect((try appendDraftChunk(std.testing.allocator, &draft, "", 0)) == null);
    try std.testing.expectEqual(@as(usize, 0), draft.buffer.items.len);
}

test "appendDraftChunk keeps whitespace-only drafts local" {
    var draft: DraftState = .{ .draft_id = 7 };
    defer draft.deinit(std.testing.allocator);

    try std.testing.expect((try appendDraftChunk(std.testing.allocator, &draft, "   \n\t", DRAFT_FLUSH_MIN_INTERVAL_MS)) == null);
    try std.testing.expectEqual(@as(usize, 5), draft.buffer.items.len);
    try std.testing.expectEqual(@as(usize, 0), draft.last_flush_len);
}

test "appendDraftChunk flushes visible content after interval" {
    var draft: DraftState = .{ .draft_id = 3 };
    defer draft.deinit(std.testing.allocator);

    const flush = (try appendDraftChunk(std.testing.allocator, &draft, "hello", DRAFT_FLUSH_MIN_INTERVAL_MS + 1)) orelse
        return error.TestUnexpectedResult;
    defer {
        var tmp = flush;
        tmp.deinit(std.testing.allocator);
    }

    try std.testing.expectEqual(@as(u64, 3), flush.draft_id);
    try std.testing.expectEqualStrings("hello", flush.text);
    try std.testing.expectEqual(@as(usize, 5), draft.last_flush_len);
}
