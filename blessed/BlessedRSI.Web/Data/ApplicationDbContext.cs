using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Data;

public class ApplicationDbContext : IdentityDbContext<ApplicationUser>
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
        : base(options)
    {
    }

    public DbSet<BacktestResult> BacktestResults { get; set; }
    public DbSet<StrategyShare> StrategyShares { get; set; }
    public DbSet<StrategyComment> StrategyComments { get; set; }
    public DbSet<CommunityPost> CommunityPosts { get; set; }
    public DbSet<PostComment> PostComments { get; set; }
    public DbSet<PostLike> PostLikes { get; set; }
    public DbSet<Achievement> Achievements { get; set; }
    public DbSet<UserAchievement> UserAchievements { get; set; }
    public DbSet<RefreshToken> RefreshTokens { get; set; }
    public DbSet<SecurityEvent> SecurityEvents { get; set; }
    public DbSet<TwoFactorAuthenticationCode> TwoFactorAuthenticationCodes { get; set; }
    public DbSet<TwoFactorBackupCode> TwoFactorBackupCodes { get; set; }
    public DbSet<TwoFactorSettings> TwoFactorSettings { get; set; }
    public DbSet<UserApiKey> UserApiKeys { get; set; }
    public DbSet<ApiKeyUsageLog> ApiKeyUsageLogs { get; set; }
    public DbSet<RateLimitEntry> RateLimitEntries { get; set; }

    protected override void OnModelCreating(ModelBuilder builder)
    {
        base.OnModelCreating(builder);

        // Configure composite keys
        builder.Entity<PostLike>()
            .HasKey(pl => new { pl.UserId, pl.PostId });

        builder.Entity<UserAchievement>()
            .HasKey(ua => new { ua.UserId, ua.AchievementId });

        // Configure relationships
        builder.Entity<BacktestResult>()
            .HasOne(br => br.User)
            .WithMany(u => u.BacktestResults)
            .HasForeignKey(br => br.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.Entity<StrategyShare>()
            .HasOne(ss => ss.BacktestResult)
            .WithOne()
            .HasForeignKey<StrategyShare>(ss => ss.BacktestResultId);

        // Configure refresh token relationships
        builder.Entity<RefreshToken>()
            .HasOne(rt => rt.User)
            .WithMany()
            .HasForeignKey(rt => rt.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.Entity<RefreshToken>()
            .HasIndex(rt => rt.Token)
            .IsUnique();

        // Configure security event relationships
        builder.Entity<SecurityEvent>()
            .HasOne(se => se.User)
            .WithMany()
            .HasForeignKey(se => se.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.Entity<SecurityEvent>()
            .Property(se => se.Metadata)
            .HasConversion(
                v => System.Text.Json.JsonSerializer.Serialize(v, (JsonSerializerOptions?)null),
                v => System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(v, (JsonSerializerOptions?)null) ?? new());

        // Configure indexes for performance
        builder.Entity<RefreshToken>()
            .HasIndex(rt => new { rt.UserId, rt.ExpiresAt });

        builder.Entity<SecurityEvent>()
            .HasIndex(se => new { se.UserId, se.CreatedAt });

        builder.Entity<SecurityEvent>()
            .HasIndex(se => se.EventType);

        // Configure 2FA relationships
        builder.Entity<TwoFactorAuthenticationCode>()
            .HasOne(c => c.User)
            .WithMany(u => u.TwoFactorCodes)
            .HasForeignKey(c => c.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.Entity<TwoFactorBackupCode>()
            .HasOne(bc => bc.User)
            .WithMany(u => u.BackupCodes)
            .HasForeignKey(bc => bc.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.Entity<TwoFactorSettings>()
            .HasOne(s => s.User)
            .WithOne(u => u.TwoFactorSettings)
            .HasForeignKey<TwoFactorSettings>(s => s.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        // Configure indexes for 2FA performance
        builder.Entity<TwoFactorAuthenticationCode>()
            .HasIndex(c => new { c.UserId, c.CodeType, c.ExpiresAt });

        builder.Entity<TwoFactorAuthenticationCode>()
            .HasIndex(c => c.Code);

        builder.Entity<TwoFactorBackupCode>()
            .HasIndex(bc => new { bc.UserId, bc.UsedAt });

        builder.Entity<TwoFactorBackupCode>()
            .HasIndex(bc => bc.Code);

        // Configure API key relationships
        builder.Entity<UserApiKey>()
            .HasOne(ak => ak.User)
            .WithMany()
            .HasForeignKey(ak => ak.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.Entity<ApiKeyUsageLog>()
            .HasOne(ul => ul.ApiKey)
            .WithMany(ak => ak.UsageLogs)
            .HasForeignKey(ul => ul.ApiKeyId)
            .OnDelete(DeleteBehavior.Cascade);

        // Configure indexes for API key performance
        builder.Entity<UserApiKey>()
            .HasIndex(ak => ak.ApiKeyHash)
            .IsUnique();

        builder.Entity<UserApiKey>()
            .HasIndex(ak => new { ak.UserId, ak.IsActive });

        builder.Entity<UserApiKey>()
            .HasIndex(ak => ak.Prefix);

        builder.Entity<ApiKeyUsageLog>()
            .HasIndex(ul => new { ul.ApiKeyId, ul.RequestedAt });

        builder.Entity<ApiKeyUsageLog>()
            .HasIndex(ul => ul.RequestedAt);

        // Configure rate limit relationships
        builder.Entity<RateLimitEntry>()
            .HasOne(rl => rl.User)
            .WithMany()
            .HasForeignKey(rl => rl.UserId)
            .OnDelete(DeleteBehavior.SetNull);

        // Configure indexes for rate limiting performance
        builder.Entity<RateLimitEntry>()
            .HasIndex(rl => new { rl.IpAddress, rl.Endpoint, rl.UserId, rl.RateLimitType, rl.WindowStart })
            .IsUnique();

        builder.Entity<RateLimitEntry>()
            .HasIndex(rl => rl.LastRequest);

        builder.Entity<RateLimitEntry>()
            .HasIndex(rl => new { rl.IpAddress, rl.RateLimitType });

        // Seed data
        builder.Entity<Achievement>().HasData(
            new Achievement
            {
                Id = 1,
                Name = "Blessed Beginner",
                Description = "Complete your first profitable backtest",
                IconClass = "fas fa-seedling",
                Type = AchievementType.FirstBacktest,
                RequiredValue = 0.01m,
                Points = 10
            },
            new Achievement
            {
                Id = 2,
                Name = "Faithful Steward",
                Description = "Achieve >70% win rate in a backtest",
                IconClass = "fas fa-crown",
                Type = AchievementType.WinRate,
                RequiredValue = 0.70m,
                Points = 25
            },
            new Achievement
            {
                Id = 3,
                Name = "David's Courage",
                Description = "Achieve Sortino ratio >5.0",
                IconClass = "fas fa-shield-alt",
                Type = AchievementType.SortinoRatio,
                RequiredValue = 5.0m,
                Points = 50
            },
            new Achievement
            {
                Id = 4,
                Name = "Solomon's Wisdom",
                Description = "Achieve Sortino ratio >8.0",
                IconClass = "fas fa-brain",
                Type = AchievementType.SortinoRatio,
                RequiredValue = 8.0m,
                Points = 100
            }
        );
    }
}