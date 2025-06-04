using Microsoft.EntityFrameworkCore;
using BlessedRSI.Web.Data;
using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Services;

public class CommunityService
{
    private readonly ApplicationDbContext _context;
    private readonly ContentSanitizationService _sanitizationService;
    private readonly ILogger<CommunityService> _logger;

    public CommunityService(
        ApplicationDbContext context, 
        ContentSanitizationService sanitizationService,
        ILogger<CommunityService> logger)
    {
        _context = context;
        _sanitizationService = sanitizationService;
        _logger = logger;
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
        try
        {
            // Sanitize content before saving
            var titleResult = _sanitizationService.SanitizeHtml(post.Title, ContentType.PlainText);
            var contentResult = _sanitizationService.SanitizeHtml(post.Content, ContentType.RichText);
            
            if (!titleResult.IsValid || !contentResult.IsValid)
            {
                _logger.LogWarning("Post creation blocked due to security concerns for user {UserId}", post.UserId);
                return null;
            }

            post.Title = titleResult.SanitizedContent;
            post.Content = contentResult.SanitizedContent;

            // Sanitize related verse if present
            if (!string.IsNullOrEmpty(post.RelatedVerse))
            {
                var verseResult = _sanitizationService.SanitizeHtml(post.RelatedVerse, ContentType.BiblicalQuote);
                if (!verseResult.IsValid)
                {
                    _logger.LogWarning("Post creation blocked due to unsafe biblical quote for user {UserId}", post.UserId);
                    return null;
                }
                post.RelatedVerse = verseResult.SanitizedContent;
            }

            // Log if content was modified
            if (titleResult.IsModified || contentResult.IsModified)
            {
                _logger.LogInformation("Post content was sanitized for user {UserId}", post.UserId);
            }

            _context.CommunityPosts.Add(post);
            await _context.SaveChangesAsync();
            
            return await _context.CommunityPosts
                .Include(p => p.User)
                .FirstOrDefaultAsync(p => p.Id == post.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating post for user {UserId}", post.UserId);
            return null;
        }
    }

    public async Task<PostComment?> AddCommentAsync(int postId, string userId, string content)
    {
        try
        {
            // Sanitize comment content
            var contentResult = _sanitizationService.SanitizeHtml(content, ContentType.UserComment);
            
            if (!contentResult.IsValid)
            {
                _logger.LogWarning("Comment creation blocked due to security concerns for user {UserId} on post {PostId}", userId, postId);
                return null;
            }

            var comment = new PostComment
            {
                PostId = postId,
                UserId = userId,
                Content = contentResult.SanitizedContent
            };

            // Log if content was modified
            if (contentResult.IsModified)
            {
                _logger.LogInformation("Comment content was sanitized for user {UserId} on post {PostId}", userId, postId);
            }

            _context.PostComments.Add(comment);
            await _context.SaveChangesAsync();

            return await _context.PostComments
                .Include(c => c.User)
                .FirstOrDefaultAsync(c => c.Id == comment.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating comment for user {UserId} on post {PostId}", userId, postId);
            return null;
        }
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