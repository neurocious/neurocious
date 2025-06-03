using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class CommunityService
{
    private readonly ApplicationDbContext _context;

    public CommunityService(ApplicationDbContext context)
    {
        _context = context;
    }

    public async Task<List<CommunityPost>> GetRecentPostsAsync(int take = 20)
    {
        return await _context.CommunityPosts
            .Include(p => p.User)
            .Include(p => p.Comments)
                .ThenInclude(c => c.User)
            .OrderByDescending(p => p.IsPinned)
            .ThenByDescending(p => p.CreatedAt)
            .Take(take)
            .ToListAsync();
    }

    public async Task<CommunityPost?> CreatePostAsync(CommunityPost post)
    {
        _context.CommunityPosts.Add(post);
        await _context.SaveChangesAsync();
        
        return await _context.CommunityPosts
            .Include(p => p.User)
            .FirstOrDefaultAsync(p => p.Id == post.Id);
    }

    public async Task<PostComment?> AddCommentAsync(int postId, string userId, string content)
    {
        var comment = new PostComment
        {
            PostId = postId,
            UserId = userId,
            Content = content
        };

        _context.PostComments.Add(comment);
        await _context.SaveChangesAsync();

        return await _context.PostComments
            .Include(c => c.User)
            .FirstOrDefaultAsync(c => c.Id == comment.Id);
    }

    public async Task<bool> ToggleLikeAsync(int postId, string userId)
    {
        var existingLike = await _context.PostLikes
            .FirstOrDefaultAsync(pl => pl.PostId == postId && pl.UserId == userId);

        if (existingLike != null)
        {
            _context.PostLikes.Remove(existingLike);
            
            var post = await _context.CommunityPosts.FindAsync(postId);
            if (post != null && post.Likes > 0)
            {
                post.Likes--;
            }
        }
        else
        {
            _context.PostLikes.Add(new PostLike
            {
                PostId = postId,
                UserId = userId
            });

            var post = await _context.CommunityPosts.FindAsync(postId);
            if (post != null)
            {
                post.Likes++;
            }
        }

        await _context.SaveChangesAsync();
        return existingLike == null; // Returns true if liked, false if unliked
    }

    public async Task<List<ActiveMember>> GetActiveMembersAsync()
    {
        var recentUsers = await _context.Users
            .Where(u => u.CommunityPosts.Any(p => p.CreatedAt >= DateTime.UtcNow.AddDays(-7)) ||
                       u.PostComments.Any(c => c.CreatedAt >= DateTime.UtcNow.AddDays(-7)))
            .Select(u => new ActiveMember
            {
                Name = $"{u.FirstName} {u.LastName}".Trim(),
                Status = "Active this week",
                IsOnline = true // This would be implemented with SignalR for real online status
            })
            .Take(10)
            .ToListAsync();

        return recentUsers;
    }

    public async Task IncrementPostViewsAsync(int postId)
    {
        var post = await _context.CommunityPosts.FindAsync(postId);
        if (post != null)
        {
            post.Views++;
            await _context.SaveChangesAsync();
        }
    }

    public async Task<List<CommunityPost>> GetPostsByTypeAsync(PostType type, int take = 20)
    {
        return await _context.CommunityPosts
            .Include(p => p.User)
            .Include(p => p.Comments)
            .Where(p => p.Type == type)
            .OrderByDescending(p => p.CreatedAt)
            .Take(take)
            .ToListAsync();
    }

    public async Task<List<CommunityPost>> GetUserPostsAsync(string userId, int take = 10)
    {
        return await _context.CommunityPosts
            .Include(p => p.Comments)
            .Where(p => p.UserId == userId)
            .OrderByDescending(p => p.CreatedAt)
            .Take(take)
            .ToListAsync();
    }
}