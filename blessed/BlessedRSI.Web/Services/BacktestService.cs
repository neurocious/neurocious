using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class BacktestService
{
    private readonly ApplicationDbContext _context;
    private readonly TradingApiService _tradingApi;

    public BacktestService(ApplicationDbContext context, TradingApiService tradingApi)
    {
        _context = context;
        _tradingApi = tradingApi;
    }

    public async Task<BacktestResult?> RunAndSaveBacktestAsync(
        BacktestParameters parameters, 
        string userId, 
        string? strategyName = null)
    {
        var result = await _tradingApi.RunBacktestAsync(parameters);
        
        if (result != null)
        {
            result.UserId = userId;
            result.StrategyName = strategyName;
            
            _context.BacktestResults.Add(result);
            await _context.SaveChangesAsync();
            
            // Update user stats
            await UpdateUserStatsAsync(userId, result);
            
            // Check for achievements
            await CheckAchievementsAsync(userId, result);
        }
        
        return result;
    }

    private async Task UpdateUserStatsAsync(string userId, BacktestResult result)
    {
        var user = await _context.Users.FindAsync(userId);
        if (user != null)
        {
            user.TotalBacktests++;
            if (result.SortinoRatio > user.BestSortinoRatio)
            {
                user.BestSortinoRatio = result.SortinoRatio;
            }
            await _context.SaveChangesAsync();
        }
    }

    private async Task CheckAchievementsAsync(string userId, BacktestResult result)
    {
        var achievements = await _context.Achievements.ToListAsync();
        var userAchievements = await _context.UserAchievements
            .Where(ua => ua.UserId == userId)
            .Select(ua => ua.AchievementId)
            .ToListAsync();

        foreach (var achievement in achievements)
        {
            if (userAchievements.Contains(achievement.Id))
                continue;

            bool earned = achievement.Type switch
            {
                AchievementType.FirstBacktest => result.TotalReturn > achievement.RequiredValue,
                AchievementType.SortinoRatio => result.SortinoRatio >= achievement.RequiredValue,
                AchievementType.WinRate => result.WinRate >= achievement.RequiredValue,
                AchievementType.TotalReturn => result.TotalReturn >= achievement.RequiredValue,
                _ => false
            };

            if (earned)
            {
                _context.UserAchievements.Add(new UserAchievement
                {
                    UserId = userId,
                    AchievementId = achievement.Id
                });

                var user = await _context.Users.FindAsync(userId);
                if (user != null)
                {
                    user.CommunityPoints += achievement.Points;
                }
            }
        }

        await _context.SaveChangesAsync();
    }

    public async Task<List<BacktestResult>> GetUserBacktestsAsync(string userId, int take = 10)
    {
        return await _context.BacktestResults
            .Where(br => br.UserId == userId)
            .OrderByDescending(br => br.CreatedAt)
            .Take(take)
            .ToListAsync();
    }

    public async Task<BacktestResult?> GetBacktestByIdAsync(int id, string userId)
    {
        return await _context.BacktestResults
            .FirstOrDefaultAsync(br => br.Id == id && br.UserId == userId);
    }

    public async Task<bool> CanUserRunBacktestAsync(string userId)
    {
        var user = await _context.Users.FindAsync(userId);
        if (user == null) return false;

        // Check subscription limits
        if (user.SubscriptionTier == SubscriptionTier.Seeker)
        {
            var today = DateTime.Today;
            var todayBacktests = await _context.BacktestResults
                .CountAsync(br => br.UserId == userId && br.CreatedAt.Date == today);
            
            return todayBacktests < 3; // Free tier limit
        }

        return true; // Paid tiers have unlimited backtests
    }
}