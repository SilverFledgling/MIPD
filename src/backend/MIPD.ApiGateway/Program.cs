// ══════════════════════════════════════════════════════════════════
// MIPD API Gateway — ASP.NET Core + YARP + JWT Auth
//
// Thuyết minh CV 6.1 + 7.3:
//   - API Gateway (YARP reverse proxy)
//   - OAuth2.0/JWT authentication
//   - RBAC 4 roles: Admin, Physician, Pharmacist, Nurse
//   - Audit trail (ai truy cập, lúc nào, thao tác gì)
//
// Frontend (:5173) → Gateway (:5000) → PK Engine (:8000)
// Run:  dotnet run --urls "http://localhost:5000"
// ══════════════════════════════════════════════════════════════════

using System.Text;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using MIPD.ApiGateway.Middleware;
using MIPD.ApiGateway.Services;

var builder = WebApplication.CreateBuilder(args);

// ── 1. YARP Reverse Proxy ──────────────────────────────────────
builder.Services
    .AddReverseProxy()
    .LoadFromConfig(builder.Configuration.GetSection("ReverseProxy"));

// ── 2. CORS (allow React frontend) ────────────────────────────
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend", policy =>
    {
        policy
            .WithOrigins(
                "http://localhost:5173",   // Vite dev server
                "http://localhost:3000",   // Alternate dev port
                "http://localhost:5000"    // Gateway itself
            )
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials();
    });
});

// ── 3. Health Checks ───────────────────────────────────────────
builder.Services.AddHealthChecks()
    .AddUrlGroup(
        new Uri("http://localhost:8000/"),
        name: "pk-engine",
        timeout: TimeSpan.FromSeconds(5)
    );

// ── 4. Distributed Cache (Redis / InMemory fallback) ──────────
var redisConnection = builder.Configuration.GetValue<string>("Redis:ConnectionString");
if (!string.IsNullOrEmpty(redisConnection))
{
    builder.Services.AddStackExchangeRedisCache(options =>
    {
        options.Configuration = redisConnection;
        options.InstanceName = "MIPD:";
    });
}
else
{
    builder.Services.AddDistributedMemoryCache();
}
builder.Services.AddSingleton<CacheService>();

// ── 5. JWT Authentication (CV 7.3) ────────────────────────────
var jwtSecret = builder.Configuration["Jwt:Secret"]
    ?? "MIPD-Default-Secret-Key-Change-In-Production-2026!";
var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? "MIPD";
var jwtAudience = builder.Configuration["Jwt:Audience"] ?? "MIPD-Client";

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = jwtIssuer,
        ValidAudience = jwtAudience,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret)),
        ClockSkew = TimeSpan.FromMinutes(1)
    };
});
builder.Services.AddAuthorization();

// ── 6. Auth + Audit Services (SRP) ────────────────────────────
builder.Services.AddSingleton<JwtService>();
builder.Services.AddSingleton<AuthService>();
builder.Services.AddSingleton<AuditService>();

var app = builder.Build();

// ── Middleware Pipeline ────────────────────────────────────────
app.UseCors("AllowFrontend");
app.UseAuthentication();    // CV 7.3: xác thực JWT
app.UseAuthorization();     // CV 7.3: phân quyền RBAC
app.UseAuditTrail();        // CV 7.3: audit trail

// ── Public Endpoints ───────────────────────────────────────────

// Health endpoint
app.MapHealthChecks("/health");

// Gateway info endpoint
app.MapGet("/", () => new
{
    service = "MIPD API Gateway",
    version = "2.1.0",
    status = "running",
    auth = "JWT Bearer",
    audit = "enabled",
    cache = !string.IsNullOrEmpty(redisConnection) ? "Redis" : "InMemory",
    routes = new
    {
        pk = "/api/pk/* → PK Engine",
        bayesian = "/api/bayesian/* → PK Engine",
        dosing = "/api/dosing/* → PK Engine",
        ai = "/api/ai/* → PK Engine",
        health = "/health",
        auth_login = "POST /auth/login",
        auth_register = "POST /auth/register",
        auth_me = "GET /auth/me [Authorized]",
        audit_recent = "GET /audit/recent [Admin]"
    }
});

// ── Auth Endpoints ─────────────────────────────────────────────

// POST /auth/login — Đăng nhập, trả JWT token
app.MapPost("/auth/login", (LoginRequest request, AuthService authService) =>
{
    var result = authService.Login(request);
    return result is not null
        ? Results.Ok(result)
        : Results.Unauthorized();
});

// POST /auth/register — Đăng ký tài khoản mới
app.MapPost("/auth/register", (RegisterRequest request, AuthService authService) =>
{
    var (response, error) = authService.Register(request);
    return response is not null
        ? Results.Ok(response)
        : Results.BadRequest(new { error });
});

// GET /auth/me — Lấy thông tin người dùng hiện tại (yêu cầu JWT)
app.MapGet("/auth/me", (HttpContext ctx, AuthService authService) =>
{
    var userId = JwtService.GetUserId(ctx.User);
    if (userId is null) return Results.Unauthorized();

    var user = authService.GetUserById(userId);
    if (user is null) return Results.NotFound();

    return Results.Ok(new
    {
        userId = user.UserId,
        email = user.Email,
        fullName = user.FullName,
        role = user.Role
    });
}).RequireAuthorization();

// ── Audit Endpoints (CV 7.3) ───────────────────────────────────

// GET /audit/recent — Xem audit log gần đây (chỉ Admin)
app.MapGet("/audit/recent", (AuditService auditService, HttpContext ctx) =>
{
    var role = JwtService.GetRole(ctx.User);
    if (role != "Admin")
        return Results.Forbid();

    var entries = auditService.GetRecent(100);
    return Results.Ok(entries);
}).RequireAuthorization();

// ── YARP reverse proxy (must be last) ──────────────────────────
app.MapReverseProxy();

app.Run();
