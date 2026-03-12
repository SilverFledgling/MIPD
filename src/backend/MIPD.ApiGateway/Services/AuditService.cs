// ══════════════════════════════════════════════════════════════════
// MIPD Audit Service — Ghi nhận hoạt động người dùng
// SRP: Chỉ quản lý audit log (ghi, truy vấn)
//
// Theo thuyết minh CV 7.3:
//   "hệ thống triển khai audit trail để ghi nhận toàn bộ
//    hoạt động người dùng (ai truy cập, truy cập lúc nào, thao tác gì)"
//
// Tạm dùng in-memory store cho dev.
// Production: ghi vào bảng AuditLog trong SQL Server (schema.sql)
// ══════════════════════════════════════════════════════════════════

using System.Collections.Concurrent;

namespace MIPD.ApiGateway.Services;

/// <summary>
/// Một bản ghi audit — ai làm gì, lúc nào, ở đâu.
/// Map trực tiếp tới bảng AuditLog trong schema.sql
/// </summary>
public record AuditEntry
{
    public long Id { get; init; }
    public string? UserId { get; init; }
    public string? UserEmail { get; init; }
    public string Action { get; init; } = "";        // HTTP method: GET, POST, PUT, DELETE
    public string EntityType { get; init; } = "";     // Route/endpoint path
    public string? EntityId { get; init; }
    public string? Details { get; init; }             // Request summary
    public string? IpAddress { get; init; }
    public int StatusCode { get; init; }
    public long DurationMs { get; init; }
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;
}

/// <summary>
/// Audit service — ghi nhận và truy vấn audit trail.
/// CV 7.3: "ai truy cập, truy cập lúc nào, thao tác gì"
/// </summary>
public class AuditService
{
    private readonly ILogger<AuditService> _logger;
    private static long _nextId;

    // In-memory store (dev). Production: SQL Server AuditLog table.
    private static readonly ConcurrentQueue<AuditEntry> _entries = new();
    private const int MaxEntries = 10_000; // Giới hạn bộ nhớ dev

    public AuditService(ILogger<AuditService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Ghi một audit entry mới.
    /// </summary>
    public void Log(AuditEntry entry)
    {
        var id = Interlocked.Increment(ref _nextId);
        var withId = entry with { Id = id };
        _entries.Enqueue(withId);

        // Giới hạn bộ nhớ
        while (_entries.Count > MaxEntries)
            _entries.TryDequeue(out _);

        _logger.LogInformation(
            "AUDIT [{Method}] {Path} by {UserId} from {IP} → {Status} ({Duration}ms)",
            entry.Action,
            entry.EntityType,
            entry.UserId ?? "anonymous",
            entry.IpAddress,
            entry.StatusCode,
            entry.DurationMs
        );
    }

    /// <summary>
    /// Truy vấn audit log — lấy N bản ghi mới nhất.
    /// </summary>
    public IEnumerable<AuditEntry> GetRecent(int count = 50)
        => _entries.Reverse().Take(count);

    /// <summary>
    /// Truy vấn audit log theo userId.
    /// </summary>
    public IEnumerable<AuditEntry> GetByUser(string userId, int count = 50)
        => _entries.Where(e => e.UserId == userId).Reverse().Take(count);
}
