using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class LeaderboardService
{
    private readonly ApplicationDbContext _context;

    public LeaderboardService(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<List<LeaderboardEntry>> GetSortinoLeadersAsync(int take = 25)
    {
        var leaders = await _context.Users
            .Where(u => u.BacktestResults.Any())
            .Select(u => new LeaderboardEntry
            {
                UserName = $"{u.FirstName} {u.LastName}".Trim(),
                FavoriteVerse = u.FavoriteVerse,
                BestStrategyName = u.BacktestResults
                    .OrderByDescending(br => br.SortinoRatio)
                    .First().StrategyName,
                BestSortinoRatio = u.BacktestResults.Max(br => br.SortinoRatio),
                BestTotalReturn = u.BacktestResults.Max(br => br.TotalReturn),
                BestWinRate = u.BacktestResults.Max(br => br.WinRate),
                CommunityPoints = u.CommunityPoints,
                LastActive = u.BacktestResults.Max(br => br.CreatedAt),
                TopAchievements = u.UserAchievements
                    .OrderByDescending(ua => ua.Achievement.Points)
                    .Take(3)
                    .Select(ua => ua.Achievement)
                    .ToList()
            })
            .OrderByDescending(le => le.BestSortinoRatio)
            .Take(take)
            .ToListAsync();

        return leaders;
    }

    public async Task<List<LeaderboardEntry>> GetReturnLeadersAsync(int take = 25)
    {
        var leaders = await _context.Users
            .Where(u => u.BacktestResults.Any())
            .Select(u => new LeaderboardEntry
            {
                UserName = $"{u.FirstName} {u.LastName}".Trim(),
                FavoriteVerse = u.FavoriteVerse,
                BestStrategyName = u.BacktestResults
                    .OrderByDescending(br => br.TotalReturn)
                    .First().StrategyName,
                BestSortinoRatio = u.BacktestResults.Max(br => br.SortinoRatio),
                BestTotalReturn = u.BacktestResults.Max(br => br.TotalReturn),
                BestWinRate = u.BacktestResults.Max(br => br.WinRate),
                CommunityPoints = u.CommunityPoints,
                LastActive = u.BacktestResults.Max(br => br.CreatedAt),
                TopAchievements = u.UserAchievements
                    .OrderByDescending(ua => ua.Achievement.Points)
                    .Take(3)
                    .Select(ua => ua.Achievement)
                    .ToList()
            })
            .OrderByDescending(le => le.BestTotalReturn)
            .Take(take)
            .ToListAsync();

        return leaders;
    }

    public async Task<List<LeaderboardEntry>> GetConsistencyLeadersAsync(int take = 25)
    {
        var leaders = await _context.Users
            .Where(u => u.BacktestResults.Any())
            .Select(u => new LeaderboardEntry
            {
                UserName = $"{u.FirstName} {u.LastName}".Trim(),
                FavoriteVerse = u.FavoriteVerse,
                BestStrategyName = u.BacktestResults
                    .OrderByDescending(br => br.WinRate)
                    .First().StrategyName,
                BestSortinoRatio = u.BacktestResults.Max(br => br.SortinoRatio),
                BestTotalReturn = u.BacktestResults.Max(br => br.TotalReturn),
                BestWinRate = u.BacktestResults.Max(br => br.WinRate),
                CommunityPoints = u.CommunityPoints,
                LastActive = u.BacktestResults.Max(br => br.CreatedAt),
                TopAchievements = u.UserAchievements
                    .OrderByDescending(ua => ua.Achievement.Points)
                    .Take(3)
                    .Select(ua => ua.Achievement)
                    .ToList()
            })
            .OrderByDescending(le => le.BestWinRate)
            .Take(take)
            .ToListAsync();

        return leaders;
    }

    public async Task<List<LeaderboardEntry>> GetCommunityLeadersAsync(int take = 25)
    {
        var leaders = await _context.Users
            .Where(u => u.CommunityPoints > 0)
            .Select(u => new LeaderboardEntry
            {
                UserName = $"{u.FirstName} {u.LastName}".Trim(),
                FavoriteVerse = u.FavoriteVerse,
                BestStrategyName = u.BacktestResults.Any() 
                    ? u.BacktestResults.OrderByDescending(br => br.SortinoRatio).First().StrategyName 
                    : null,
                BestSortinoRatio = u.BacktestResults.Any() ? u.BacktestResults.Max(br => br.SortinoRatio) : 0,
                BestTotalReturn = u.BacktestResults.Any() ? u.BacktestResults.Max(br => br.TotalReturn) : 0,
                BestWinRate = u.BacktestResults.Any() ? u.BacktestResults.Max(br => br.WinRate) : 0,
                CommunityPoints = u.CommunityPoints,
                LastActive = u.CommunityPosts.Any() 
                    ? u.CommunityPosts.Max(cp => cp.CreatedAt) 
                    : u.BacktestResults.Any() 
                        ? u.BacktestResults.Max(br => br.CreatedAt) 
                        : u.CreatedAt,
                TopAchievements = u.UserAchievements
                    .OrderByDescending(ua => ua.Achievement.Points)
                    .Take(3)
                    .Select(ua => ua.Achievement)
                    .ToList()
            })
            .OrderByDescending(le => le.CommunityPoints)
            .Take(take)
            .ToListAsync();

        return leaders;
    }

    public async Task<List<FeaturedStrategy>> GetFeaturedStrategiesAsync()
    {
        var strategies = await _context.StrategyShares
            .Include(ss => ss.User)
            .Include(ss => ss.BacktestResult)
            .Where(ss => ss.BacktestResult.SortinoRatio > 3.0m)
            .OrderByDescending(ss => ss.Likes)
            .ThenByDescending(ss => ss.BacktestResult.SortinoRatio)
            .Take(5)
            .Select(ss => new FeaturedStrategy
            {
                Id = ss.Id,
                BiblicalName = ss.BiblicalName,
                CreatedBy = $"{ss.User.FirstName} {ss.User.LastName}".Trim(),
                SortinoRatio = ss.BacktestResult.SortinoRatio,
                RelatedVerse = ss.RelatedVerse
            })
            .ToListAsync();

        return strategies;
    }

    public async Task<List<RecentAchievement>> GetRecentAchievementsAsync()
    {
        var achievements = await _context.UserAchievements
            .Include(ua => ua.User)
            .Include(ua => ua.Achievement)
            .Where(ua => ua.EarnedAt >= DateTime.UtcNow.AddDays(-30))
            .OrderByDescending(ua => ua.EarnedAt)
            .Take(10)
            .Select(ua => new RecentAchievement
            {
                UserName = $"{ua.User.FirstName} {ua.User.LastName}".Trim(),
                AchievementName = ua.Achievement.Name,
                IconClass = ua.Achievement.IconClass
            })
            .ToListAsync();

        return achievements;
    }

    public async Task<CommunityStats> GetCommunityStatsAsync()
    {
        var stats = new CommunityStats
        {
            TotalMembers = await _context.Users.CountAsync(),
            TotalBacktests = await _context.BacktestResults.CountAsync(),
            SharedStrategies = await _context.StrategyShares.CountAsync()
        };

        var avgSortino = await _context.BacktestResults
            .Where(br => br.SortinoRatio > 0)
            .AverageAsync(br => (double?)br.SortinoRatio);

        stats.AvgSortinoRatio = (decimal)(avgSortino ?? 0);

        return stats;
    }
}