// ══════════════════════════════════════════════════════════════════
// MIPD Audit Middleware — Tự động ghi nhận mọi HTTP request
// SRP: Chỉ capture request/response metadata rồi chuyển cho AuditService
//
// Theo thuyết minh CV 7.3:
//   Audit trail ghi nhận: ai truy cập, truy cập lúc nào, thao tác gì
// ══════════════════════════════════════════════════════════════════

using System.Diagnostics;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using MIPD.ApiGateway.Services;

namespace MIPD.ApiGateway.Middleware;

/// <summary>
/// Middleware tự động ghi audit log cho mọi HTTP request.
/// Capture: method, path, userId (từ JWT), IP, status code, duration.
/// </summary>
public class AuditMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<AuditMiddleware> _logger;

    // Bỏ qua các path không cần audit (health check, static files)
    private static readonly HashSet<string> SkipPaths = new(StringComparer.OrdinalIgnoreCase)
    {
        "/health",
        "/favicon.ico"
    };

    public AuditMiddleware(RequestDelegate next, ILogger<AuditMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context, AuditService auditService)
    {
        var path = context.Request.Path.Value ?? "/";

        // Bỏ qua health check và static files
        if (SkipPaths.Contains(path))
        {
            await _next(context);
            return;
        }

        var stopwatch = Stopwatch.StartNew();

        // Xử lý request
        await _next(context);

        stopwatch.Stop();

        // Lấy userId từ JWT claims (nếu đã xác thực)
        var userId = context.User.FindFirst(JwtRegisteredClaimNames.Sub)?.Value
                     ?? context.User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        var userEmail = context.User.FindFirst(JwtRegisteredClaimNames.Email)?.Value
                        ?? context.User.FindFirst(ClaimTypes.Email)?.Value;

        // Lấy IP address
        var ipAddress = context.Connection.RemoteIpAddress?.ToString();

        // Ghi audit entry
        var entry = new AuditEntry
        {
            UserId = userId,
            UserEmail = userEmail,
            Action = context.Request.Method,            // GET, POST, PUT, DELETE
            EntityType = path,                          // /api/pk/predict, /auth/login, etc.
            EntityId = context.Request.QueryString.HasValue
                ? context.Request.QueryString.Value
                : null,
            Details = $"{context.Request.Method} {path} → {context.Response.StatusCode}",
            IpAddress = ipAddress,
            StatusCode = context.Response.StatusCode,
            DurationMs = stopwatch.ElapsedMilliseconds,
            Timestamp = DateTime.UtcNow
        };

        auditService.Log(entry);
    }
}

/// <summary>
/// Extension method để đăng ký AuditMiddleware.
/// </summary>
public static class AuditMiddlewareExtensions
{
    public static IApplicationBuilder UseAuditTrail(this IApplicationBuilder builder)
        => builder.UseMiddleware<AuditMiddleware>();
}
