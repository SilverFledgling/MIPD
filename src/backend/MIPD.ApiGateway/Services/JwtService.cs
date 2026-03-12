// ══════════════════════════════════════════════════════════════════
// MIPD JWT Service — Token generation & validation
// SRP: Chỉ quản lý JWT token (tạo, đọc claims)
//
// Theo thuyết minh CV 7.3:
//   - OAuth2.0/OpenID Connect + JWT token
//   - Quyền truy cập riêng biệt cho từng vai trò (bác sĩ, dược sĩ, quản trị viên)
// ══════════════════════════════════════════════════════════════════

using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;

namespace MIPD.ApiGateway.Services;

/// <summary>
/// JWT token service — tạo và validate access token.
/// </summary>
public class JwtService
{
    private readonly IConfiguration _config;
    private readonly ILogger<JwtService> _logger;

    public JwtService(IConfiguration config, ILogger<JwtService> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Tạo JWT access token cho user đã xác thực.
    /// Claims: sub (userId), email, name, role
    /// </summary>
    public string GenerateToken(string userId, string email, string fullName, string role)
    {
        var jwtSettings = _config.GetSection("Jwt");
        var key = new SymmetricSecurityKey(
            Encoding.UTF8.GetBytes(jwtSettings["Secret"] ?? throw new InvalidOperationException("JWT Secret not configured"))
        );

        var claims = new[]
        {
            new Claim(JwtRegisteredClaimNames.Sub, userId),
            new Claim(JwtRegisteredClaimNames.Email, email),
            new Claim(JwtRegisteredClaimNames.Name, fullName),
            new Claim(ClaimTypes.Role, role),
            new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString()),
            new Claim(JwtRegisteredClaimNames.Iat,
                DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString(),
                ClaimValueTypes.Integer64)
        };

        var expireMinutes = int.Parse(jwtSettings["ExpireMinutes"] ?? "480"); // Default 8h

        var token = new JwtSecurityToken(
            issuer: jwtSettings["Issuer"] ?? "MIPD",
            audience: jwtSettings["Audience"] ?? "MIPD-Client",
            claims: claims,
            expires: DateTime.UtcNow.AddMinutes(expireMinutes),
            signingCredentials: new SigningCredentials(key, SecurityAlgorithms.HmacSha256)
        );

        var tokenString = new JwtSecurityTokenHandler().WriteToken(token);
        _logger.LogInformation("JWT issued for {Email} (role: {Role})", email, role);
        return tokenString;
    }

    /// <summary>
    /// Đọc UserId từ ClaimsPrincipal (sau khi middleware validate token).
    /// </summary>
    public static string? GetUserId(ClaimsPrincipal user)
        => user.FindFirst(JwtRegisteredClaimNames.Sub)?.Value
           ?? user.FindFirst(ClaimTypes.NameIdentifier)?.Value;

    /// <summary>
    /// Đọc Role từ ClaimsPrincipal.
    /// </summary>
    public static string? GetRole(ClaimsPrincipal user)
        => user.FindFirst(ClaimTypes.Role)?.Value;
}
