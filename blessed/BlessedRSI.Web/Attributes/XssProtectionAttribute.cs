using System.ComponentModel.DataAnnotations;
using BlessedRSI.Web.Services;

namespace BlessedRSI.Web.Attributes;

[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter)]
public class XssProtectionAttribute : ValidationAttribute
{
    private readonly ContentType _contentType;
    private readonly bool _allowHtml;

    public XssProtectionAttribute(ContentType contentType = ContentType.PlainText, bool allowHtml = false)
    {
        _contentType = contentType;
        _allowHtml = allowHtml;
    }

    protected override ValidationResult? IsValid(object? value, ValidationContext validationContext)
    {
        if (value == null || string.IsNullOrWhiteSpace(value.ToString()))
        {
            return ValidationResult.Success;
        }

        var content = value.ToString()!;
        
        // Get the sanitization service from DI
        var sanitizationService = validationContext.GetService(typeof(ContentSanitizationService)) as ContentSanitizationService;
        
        if (sanitizationService == null)
        {
            // Fallback validation if service is not available
            return ValidateFallback(content, validationContext);
        }

        try
        {
            // Check if content is safe
            if (!sanitizationService.IsContentSafe(content))
            {
                return new ValidationResult(
                    "Content contains potentially dangerous elements and cannot be accepted.",
                    new[] { validationContext.MemberName ?? "Content" });
            }

            // For strict validation, ensure no HTML in plain text fields
            if (_contentType == ContentType.PlainText && !_allowHtml)
            {
                var containsHtml = System.Text.RegularExpressions.Regex.IsMatch(content, @"<[^>]+>");
                if (containsHtml)
                {
                    return new ValidationResult(
                        "HTML content is not allowed in this field.",
                        new[] { validationContext.MemberName ?? "Content" });
                }
            }

            return ValidationResult.Success;
        }
        catch (Exception)
        {
            // Fallback on error
            return ValidateFallback(content, validationContext);
        }
    }

    private ValidationResult? ValidateFallback(string content, ValidationContext validationContext)
    {
        // Basic XSS pattern detection
        var dangerousPatterns = new[]
        {
            @"<script[^>]*>.*?</script>",
            @"javascript:",
            @"vbscript:",
            @"on\w+\s*=",
            @"<iframe[^>]*>",
            @"<object[^>]*>",
            @"<embed[^>]*>"
        };

        foreach (var pattern in dangerousPatterns)
        {
            if (System.Text.RegularExpressions.Regex.IsMatch(content, pattern, 
                System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            {
                return new ValidationResult(
                    "Content contains potentially dangerous elements.",
                    new[] { validationContext.MemberName ?? "Content" });
            }
        }

        return ValidationResult.Success;
    }
}

[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
public class SafeHtmlAttribute : XssProtectionAttribute
{
    public SafeHtmlAttribute() : base(ContentType.RichText, true)
    {
    }
}

[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
public class PlainTextOnlyAttribute : XssProtectionAttribute
{
    public PlainTextOnlyAttribute() : base(ContentType.PlainText, false)
    {
    }
}

[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
public class BiblicalQuoteAttribute : XssProtectionAttribute
{
    public BiblicalQuoteAttribute() : base(ContentType.BiblicalQuote, false)
    {
    }
}