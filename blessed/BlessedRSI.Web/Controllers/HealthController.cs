using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Controllers;

[ApiController]
[Route("api/[controller]")]
public class HealthController : ControllerBase
{
    private readonly ApplicationDbContext _context;
    private readonly EnhancedEmailService _emailService;
    private readonly ILogger<HealthController> _logger;

    public HealthController(
        ApplicationDbContext context,
        EnhancedEmailService emailService,
        ILogger<HealthController> logger)
    {
        _context = context;
        _emailService = emailService;
        _logger = logger;
    }

    [HttpGet]
    public async Task<ActionResult> GetHealth()
    {
        var healthData = new
        {
            status = "healthy",
            timestamp = DateTime.UtcNow,
            version = "1.0.0",
            environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Unknown",
            uptime = GetUptime(),
            biblical = new
            {
                verse = "\"He gives strength to the weary and increases the power of the weak.\"",
                reference = "Isaiah 40:29"
            }
        };

        return Ok(healthData);
    }

    [HttpGet("detailed")]
    public async Task<ActionResult> GetDetailedHealth()
    {
        var checks = new Dictionary<string, object>();
        var overallHealthy = true;

        try
        {
            // Database health check
            var dbHealth = await CheckDatabaseHealthAsync();
            checks["database"] = dbHealth;
            if (!(bool)((dynamic)dbHealth).healthy)
                overallHealthy = false;

            // Email health check
            var emailHealth = await CheckEmailHealthAsync();
            checks["email"] = emailHealth;
            if (!(bool)((dynamic)emailHealth).healthy)
                overallHealthy = false;

            // Memory health check
            var memoryHealth = CheckMemoryHealth();
            checks["memory"] = memoryHealth;

            // Disk health check
            var diskHealth = CheckDiskHealth();
            checks["disk"] = diskHealth;

            return Ok(new
            {
                status = overallHealthy ? "healthy" : "degraded",
                timestamp = DateTime.UtcNow,
                checks = checks,
                overall = new
                {
                    healthy = overallHealthy,
                    environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Unknown",
                    version = "1.0.0",
                    uptime = GetUptime()
                },
                biblical = new
                {
                    verse = overallHealthy 
                        ? "\"The Lord your God is with you, the Mighty Warrior who saves.\""
                        : "\"The Lord is close to the brokenhearted and saves those who are crushed in spirit.\"",
                    reference = overallHealthy ? "Zephaniah 3:17" : "Psalm 34:18"
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error performing detailed health check");
            return StatusCode(500, new
            {
                status = "unhealthy",
                timestamp = DateTime.UtcNow,
                error = "Health check failed",
                message = ex.Message
            });
        }
    }

    [HttpGet("ping")]
    public ActionResult Ping()
    {
        return Ok(new
        {
            message = "pong",
            timestamp = DateTime.UtcNow,
            server = Environment.MachineName
        });
    }

    private async Task<object> CheckDatabaseHealthAsync()
    {
        try
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            
            // Test connection
            var canConnect = await _context.Database.CanConnectAsync();
            if (!canConnect)
            {
                return new
                {
                    healthy = false,
                    message = "Cannot connect to database",
                    responseTime = stopwatch.ElapsedMilliseconds
                };
            }

            // Test a simple query
            var userCount = await _context.Users.CountAsync();
            stopwatch.Stop();

            // Check for pending migrations
            var pendingMigrations = await _context.Database.GetPendingMigrationsAsync();

            return new
            {
                healthy = true,
                message = "Database connection successful",
                responseTime = stopwatch.ElapsedMilliseconds,
                details = new
                {
                    userCount = userCount,
                    pendingMigrations = pendingMigrations.Count(),
                    connectionState = "Open"
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Database health check failed");
            return new
            {
                healthy = false,
                message = "Database health check failed",
                error = ex.Message,
                responseTime = -1
            };
        }
    }

    private async Task<object> CheckEmailHealthAsync()
    {
        try
        {
            var primaryHealth = await _emailService.GetHealthStatusAsync(EmailProvider.Primary);
            var fallbackHealth = await _emailService.GetHealthStatusAsync(EmailProvider.Fallback);

            var isHealthy = primaryHealth.IsHealthy || fallbackHealth.IsHealthy;

            return new
            {
                healthy = isHealthy,
                message = isHealthy ? "Email service operational" : "Email service issues detected",
                details = new
                {
                    primary = new
                    {
                        healthy = primaryHealth.IsHealthy,
                        lastSuccessful = primaryHealth.LastSuccessful,
                        consecutiveFailures = primaryHealth.ConsecutiveFailures,
                        lastError = primaryHealth.LastError
                    },
                    fallback = new
                    {
                        healthy = fallbackHealth.IsHealthy,
                        available = fallbackHealth.FallbackAvailable
                    }
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Email health check failed");
            return new
            {
                healthy = false,
                message = "Email health check failed",
                error = ex.Message
            };
        }
    }

    private object CheckMemoryHealth()
    {
        try
        {
            var process = System.Diagnostics.Process.GetCurrentProcess();
            var workingSetMB = process.WorkingSet64 / 1024 / 1024;
            var privateMemoryMB = process.PrivateMemorySize64 / 1024 / 1024;

            // Consider unhealthy if using more than 1GB
            var isHealthy = workingSetMB < 1024;

            return new
            {
                healthy = isHealthy,
                message = isHealthy ? "Memory usage normal" : "High memory usage detected",
                details = new
                {
                    workingSetMB = workingSetMB,
                    privateMemoryMB = privateMemoryMB,
                    gcMemoryMB = GC.GetTotalMemory(false) / 1024 / 1024
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Memory health check failed");
            return new
            {
                healthy = false,
                message = "Memory health check failed",
                error = ex.Message
            };
        }
    }

    private object CheckDiskHealth()
    {
        try
        {
            var drives = DriveInfo.GetDrives()
                .Where(d => d.IsReady && d.DriveType == DriveType.Fixed)
                .Select(d => new
                {
                    name = d.Name,
                    totalSizeGB = d.TotalSize / 1024 / 1024 / 1024,
                    availableSpaceGB = d.AvailableFreeSpace / 1024 / 1024 / 1024,
                    freeSpacePercentage = (double)d.AvailableFreeSpace / d.TotalSize * 100
                })
                .ToList();

            // Consider unhealthy if any drive has less than 10% free space
            var isHealthy = drives.All(d => d.freeSpacePercentage > 10);

            return new
            {
                healthy = isHealthy,
                message = isHealthy ? "Disk space sufficient" : "Low disk space detected",
                details = drives
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Disk health check failed");
            return new
            {
                healthy = false,
                message = "Disk health check failed",
                error = ex.Message
            };
        }
    }

    private TimeSpan GetUptime()
    {
        return DateTime.UtcNow - System.Diagnostics.Process.GetCurrentProcess().StartTime.ToUniversalTime();
    }
}