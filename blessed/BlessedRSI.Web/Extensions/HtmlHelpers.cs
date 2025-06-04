using Microsoft.AspNetCore.Html;
using Microsoft.AspNetCore.Mvc.Rendering;
using System.Text.Encodings.Web;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Extensions;

public static class HtmlHelpers
{
    public static IHtmlContent SafeContent(this IHtmlHelper htmlHelper, string? content)
    {
        if (string.IsNullOrEmpty(content))
            return new HtmlString(string.Empty);

        // Get sanitization service from DI
        var httpContext = htmlHelper.ViewContext.HttpContext;
        var sanitizationService = httpContext.RequestServices.GetService<ContentSanitizationService>();
        
        if (sanitizationService != null)
        {
            var result = sanitizationService.SanitizeHtml(content, ContentType.RichText);
            return new HtmlString(result.SanitizedContent);
        }

        // Fallback to HTML encoding if service is not available
        return new HtmlString(HtmlEncoder.Default.Encode(content));
    }

    public static IHtmlContent SafePlainText(this IHtmlHelper htmlHelper, string? content)
    {
        if (string.IsNullOrEmpty(content))
            return new HtmlString(string.Empty);

        // Get sanitization service from DI
        var httpContext = htmlHelper.ViewContext.HttpContext;
        var sanitizationService = httpContext.RequestServices.GetService<ContentSanitizationService>();
        
        if (sanitizationService != null)
        {
            var sanitized = sanitizationService.SanitizePlainText(content);
            return new HtmlString(sanitized);
        }

        // Fallback to HTML encoding
        return new HtmlString(HtmlEncoder.Default.Encode(content));
    }

    public static IHtmlContent SafeBiblicalQuote(this IHtmlHelper htmlHelper, string? verse, string? reference = null)
    {
        if (string.IsNullOrEmpty(verse))
            return new HtmlString(string.Empty);

        // Get sanitization service from DI
        var httpContext = htmlHelper.ViewContext.HttpContext;
        var sanitizationService = httpContext.RequestServices.GetService<ContentSanitizationService>();
        
        if (sanitizationService != null && !string.IsNullOrEmpty(reference))
        {
            var safeQuote = sanitizationService.CreateBiblicalQuote(verse, reference);
            return new HtmlString(safeQuote);
        }

        // Fallback - just encode the verse
        var encodedVerse = HtmlEncoder.Default.Encode(verse);
        if (!string.IsNullOrEmpty(reference))
        {
            var encodedReference = HtmlEncoder.Default.Encode(reference);
            return new HtmlString($"<em>\"{encodedVerse}\"</em> - {encodedReference}");
        }

        return new HtmlString($"<em>\"{encodedVerse}\"</em>");
    }

    public static IHtmlContent TruncatedSafeContent(this IHtmlHelper htmlHelper, string? content, int maxLength = 200)
    {
        if (string.IsNullOrEmpty(content))
            return new HtmlString(string.Empty);

        // Get sanitization service from DI
        var httpContext = htmlHelper.ViewContext.HttpContext;
        var sanitizationService = httpContext.RequestServices.GetService<ContentSanitizationService>();
        
        string processedContent;
        if (sanitizationService != null)
        {
            var result = sanitizationService.SanitizeHtml(content, ContentType.RichText);
            processedContent = result.SanitizedContent;
        }
        else
        {
            processedContent = HtmlEncoder.Default.Encode(content);
        }

        // Truncate content intelligently
        if (processedContent.Length <= maxLength)
            return new HtmlString(processedContent);

        // Find a good breaking point
        var truncated = processedContent.Substring(0, maxLength);
        var lastSpace = truncated.LastIndexOf(' ');
        
        if (lastSpace > maxLength * 0.7) // Don't break too early
        {
            truncated = truncated.Substring(0, lastSpace);
        }

        return new HtmlString(truncated + "...");
    }

    public static string GetPostTypeIcon(PostType postType)
    {
        return postType switch
        {
            PostType.Discussion => "fas fa-comments",
            PostType.PrayerRequest => "fas fa-praying-hands",
            PostType.Testimony => "fas fa-heart",
            PostType.Question => "fas fa-question-circle",
            PostType.MarketInsight => "fas fa-chart-line",
            PostType.BiblicalReflection => "fas fa-bible",
            _ => "fas fa-comment"
        };
    }

    public static string GetPostTypeColor(PostType postType)
    {
        return postType switch
        {
            PostType.Discussion => "primary",
            PostType.PrayerRequest => "warning",
            PostType.Testimony => "success",
            PostType.Question => "info",
            PostType.MarketInsight => "danger",
            PostType.BiblicalReflection => "secondary",
            _ => "light"
        };
    }

    public static string GetPostTypeName(PostType postType)
    {
        return postType switch
        {
            PostType.Discussion => "Discussion",
            PostType.PrayerRequest => "Prayer Request",
            PostType.Testimony => "Testimony",
            PostType.Question => "Question",
            PostType.MarketInsight => "Market Insight",
            PostType.BiblicalReflection => "Biblical Reflection",
            _ => "Post"
        };
    }

    public static string TimeAgo(DateTime dateTime)
    {
        var timeSpan = DateTime.UtcNow - dateTime;

        if (timeSpan.TotalDays > 365)
            return $"{(int)(timeSpan.TotalDays / 365)} year{((int)(timeSpan.TotalDays / 365) == 1 ? "" : "s")} ago";
        
        if (timeSpan.TotalDays > 30)
            return $"{(int)(timeSpan.TotalDays / 30)} month{((int)(timeSpan.TotalDays / 30) == 1 ? "" : "s")} ago";
        
        if (timeSpan.TotalDays > 1)
            return $"{(int)timeSpan.TotalDays} day{((int)timeSpan.TotalDays == 1 ? "" : "s")} ago";
        
        if (timeSpan.TotalHours > 1)
            return $"{(int)timeSpan.TotalHours} hour{((int)timeSpan.TotalHours == 1 ? "" : "s")} ago";
        
        if (timeSpan.TotalMinutes > 1)
            return $"{(int)timeSpan.TotalMinutes} minute{((int)timeSpan.TotalMinutes == 1 ? "" : "s")} ago";
        
        return "Just now";
    }
}