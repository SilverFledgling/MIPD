// ══════════════════════════════════════════════════════════════════
// MIPD Auth Service — Login, Registration, Password hashing
// SRP: Chỉ quản lý business logic xác thực
//
// Tạm thời dùng in-memory store cho dev.
// Production: kết nối MS SQL Server (bảng Users trong schema.sql)
// ══════════════════════════════════════════════════════════════════

namespace MIPD.ApiGateway.Services;

/// <summary>
/// DTO cho request login.
/// </summary>
public record LoginRequest(string Email, string Password);

/// <summary>
/// DTO cho request register.
/// </summary>
public record RegisterRequest(string Email, string Password, string FullName, string Role);

/// <summary>
/// DTO cho response auth (token + user info).
/// </summary>
public record AuthResponse(string Token, string UserId, string Email, string FullName, string Role);

/// <summary>
/// In-memory user model (dev). Production: map to SQL Server Users table.
/// </summary>
public class UserRecord
{
    public string UserId { get; set; } = Guid.NewGuid().ToString();
    public string Email { get; set; } = "";
    public string PasswordHash { get; set; } = "";
    public string FullName { get; set; } = "";
    public string Role { get; set; } = "Physician";
    public bool IsActive { get; set; } = true;
}

/// <summary>
/// Authentication service — login, register, validate.
/// Theo thuyết minh CV 7.3: xác thực OAuth2/JWT, phân quyền 4 roles.
/// </summary>
public class AuthService
{
    private readonly JwtService _jwtService;
    private readonly ILogger<AuthService> _logger;

    // In-memory user store (dev). Production: inject IUserRepository → SQL Server.
    private static readonly List<UserRecord> _users = new()
    {
        // Default admin account
        new UserRecord
        {
            UserId = "00000000-0000-0000-0000-000000000001",
            Email = "admin@mipd.vn",
            PasswordHash = BCrypt.Net.BCrypt.HashPassword("Admin@123"),
            FullName = "System Administrator",
            Role = "Admin"
        }
    };

    private static readonly HashSet<string> ValidRoles = new()
    {
        "Admin", "Physician", "Pharmacist", "Nurse"
    };

    public AuthService(JwtService jwtService, ILogger<AuthService> logger)
    {
        _jwtService = jwtService;
        _logger = logger;
    }

    /// <summary>
    /// Đăng nhập — trả JWT token nếu xác thực thành công.
    /// </summary>
    public AuthResponse? Login(LoginRequest request)
    {
        var user = _users.FirstOrDefault(u =>
            u.Email.Equals(request.Email, StringComparison.OrdinalIgnoreCase) && u.IsActive);

        if (user is null)
        {
            _logger.LogWarning("Login failed: user {Email} not found", request.Email);
            return null;
        }

        if (!BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
        {
            _logger.LogWarning("Login failed: wrong password for {Email}", request.Email);
            return null;
        }

        var token = _jwtService.GenerateToken(user.UserId, user.Email, user.FullName, user.Role);
        _logger.LogInformation("Login success: {Email} (role: {Role})", user.Email, user.Role);

        return new AuthResponse(token, user.UserId, user.Email, user.FullName, user.Role);
    }

    /// <summary>
    /// Đăng ký tài khoản mới — trả JWT token nếu thành công.
    /// </summary>
    public (AuthResponse? Response, string? Error) Register(RegisterRequest request)
    {
        // Validate role
        if (!ValidRoles.Contains(request.Role))
            return (null, $"Invalid role. Must be: {string.Join(", ", ValidRoles)}");

        // Check duplicate email
        if (_users.Any(u => u.Email.Equals(request.Email, StringComparison.OrdinalIgnoreCase)))
            return (null, "Email already registered");

        // Validate password
        if (string.IsNullOrWhiteSpace(request.Password) || request.Password.Length < 8)
            return (null, "Password must be at least 8 characters");

        var user = new UserRecord
        {
            Email = request.Email,
            PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.Password),
            FullName = request.FullName,
            Role = request.Role
        };

        _users.Add(user);
        _logger.LogInformation("User registered: {Email} (role: {Role})", user.Email, user.Role);

        var token = _jwtService.GenerateToken(user.UserId, user.Email, user.FullName, user.Role);
        return (new AuthResponse(token, user.UserId, user.Email, user.FullName, user.Role), null);
    }

    /// <summary>
    /// Lấy thông tin user theo ID.
    /// </summary>
    public UserRecord? GetUserById(string userId)
        => _users.FirstOrDefault(u => u.UserId == userId && u.IsActive);
}
