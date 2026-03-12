// ══════════════════════════════════════════════════════════════════
// MIPD Cache Service — IDistributedCache wrapper
// SRP: Chỉ quản lý caching logic (get/set/remove)
//
// Hỗ trợ 2 backend:
//   - Redis (production)  — khi Redis connection string config
//   - InMemory (dev/test) — fallback khi không có Redis
//
// Config trong appsettings.json:
//   "Redis": { "ConnectionString": "localhost:6379" }
//   hoặc bỏ trống để dùng InMemory
// ══════════════════════════════════════════════════════════════════

using Microsoft.Extensions.Caching.Distributed;
using System.Text.Json;

namespace MIPD.ApiGateway.Services;

/// <summary>
/// Cache service cho MIPD — wrap IDistributedCache với JSON serialization.
/// Cache kết quả PK/Bayesian tính toán tốn kém để giảm tải PK Engine.
/// </summary>
public class CacheService
{
    private readonly IDistributedCache _cache;
    private readonly ILogger<CacheService> _logger;

    // Default cache durations
    private static readonly TimeSpan DefaultExpiration = TimeSpan.FromMinutes(30);
    private static readonly TimeSpan SessionExpiration = TimeSpan.FromHours(8);    // Phiên làm việc
    private static readonly TimeSpan ResultExpiration = TimeSpan.FromMinutes(15);   // Kết quả tính toán

    public CacheService(IDistributedCache cache, ILogger<CacheService> logger)
    {
        _cache = cache;
        _logger = logger;
    }

    /// <summary>
    /// Lấy giá trị từ cache, trả null nếu không tìm thấy.
    /// </summary>
    public async Task<T?> GetAsync<T>(string key, CancellationToken ct = default) where T : class
    {
        try
        {
            var data = await _cache.GetStringAsync(key, ct);
            if (data is null) return null;

            _logger.LogDebug("Cache HIT: {Key}", key);
            return JsonSerializer.Deserialize<T>(data);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Cache GET failed for key {Key}", key);
            return null;
        }
    }

    /// <summary>
    /// Lưu giá trị vào cache với thời gian hết hạn tùy chỉnh.
    /// </summary>
    public async Task SetAsync<T>(string key, T value,
        TimeSpan? expiration = null, CancellationToken ct = default) where T : class
    {
        try
        {
            var options = new DistributedCacheEntryOptions
            {
                AbsoluteExpirationRelativeToNow = expiration ?? DefaultExpiration
            };
            var json = JsonSerializer.Serialize(value);
            await _cache.SetStringAsync(key, json, options, ct);
            _logger.LogDebug("Cache SET: {Key} (TTL: {TTL})", key, expiration ?? DefaultExpiration);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Cache SET failed for key {Key}", key);
        }
    }

    /// <summary>
    /// Xóa một key khỏi cache.
    /// </summary>
    public async Task RemoveAsync(string key, CancellationToken ct = default)
    {
        try
        {
            await _cache.RemoveAsync(key, ct);
            _logger.LogDebug("Cache REMOVE: {Key}", key);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Cache REMOVE failed for key {Key}", key);
        }
    }

    // ── Key builders ──────────────────────────────────────────────

    /// <summary>
    /// Key cho kết quả PK prediction (cache 15 phút).
    /// </summary>
    public static string PkResultKey(string patientId, string drugName)
        => $"pk:result:{patientId}:{drugName}";

    /// <summary>
    /// Key cho kết quả Bayesian estimation (cache 15 phút).
    /// </summary>
    public static string BayesianResultKey(string patientId, string method)
        => $"bayesian:result:{patientId}:{method}";

    /// <summary>
    /// Key cho dosing recommendation (cache 15 phút).
    /// </summary>
    public static string DosingResultKey(string patientId)
        => $"dosing:result:{patientId}";

    /// <summary>
    /// Key cho session data (cache 8 giờ).
    /// </summary>
    public static string SessionKey(string userId)
        => $"session:{userId}";

    /// <summary>
    /// Lưu kết quả tính toán PK/Bayesian (15 phút TTL).
    /// </summary>
    public Task SetResultAsync<T>(string key, T value, CancellationToken ct = default) where T : class
        => SetAsync(key, value, ResultExpiration, ct);

    /// <summary>
    /// Lưu session data (8 giờ TTL).
    /// </summary>
    public Task SetSessionAsync<T>(string key, T value, CancellationToken ct = default) where T : class
        => SetAsync(key, value, SessionExpiration, ct);
}
