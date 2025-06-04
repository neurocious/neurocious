using System.Text.RegularExpressions;
using System.Web;
using System.Text;
using HtmlAgilityPack;

namespace BlessedRSI.Web.Services;

public class ContentSanitizationService
{
    private readonly ILogger<ContentSanitizationService> _logger;
    private readonly HashSet<string> _allowedTags;
    private readonly HashSet<string> _allowedAttributes;
    private readonly Dictionary<string, HashSet<string>> _tagSpecificAttributes;
    private readonly List<Regex> _maliciousPatterns;

    public ContentSanitizationService(ILogger<ContentSanitizationService> logger)
    {
        _logger = logger;
        
        // Allow safe HTML tags for formatting
        _allowedTags = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "b", "strong", "i", "em", "u", "p", "br", "ul", "ol", "li", 
            "blockquote", "span", "div", "h1", "h2", "h3", "h4", "h5", "h6"
        };

        // Allow safe attributes
        _allowedAttributes = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "class", "title", "alt"
        };

        // Tag-specific allowed attributes
        _tagSpecificAttributes = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase)
        {
            ["blockquote"] = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "cite", "class" },
            ["span"] = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "class", "title" },
            ["div"] = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "class" }
        };

        // Patterns for detecting malicious content
        _maliciousPatterns = new List<Regex>
        {
            new Regex(@"javascript:", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"vbscript:", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"data:", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"on\w+\s*=", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"<script[^>]*>.*?</script>", RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.Singleline),
            new Regex(@"<iframe[^>]*>.*?</iframe>", RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.Singleline),
            new Regex(@"<object[^>]*>.*?</object>", RegexOptions.IgnoreCase | RegexOptions.Compiled | RegexOptions.Singleline),
            new Regex(@"<embed[^>]*>", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"<form[^>]*>", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"<meta[^>]*>", RegexOptions.IgnoreCase | RegexOptions.Compiled)
        };
    }

    public SanitizationResult SanitizeHtml(string input, ContentType contentType = ContentType.General)
    {
        if (string.IsNullOrEmpty(input))
        {
            return new SanitizationResult
            {
                SanitizedContent = string.Empty,
                IsModified = false,
                IsValid = true
            };
        }

        var result = new SanitizationResult
        {
            OriginalContent = input
        };

        try
        {
            // Step 1: Detect obviously malicious patterns
            var threatLevel = DetectThreatLevel(input);
            result.ThreatLevel = threatLevel;

            if (threatLevel == ThreatLevel.High)
            {
                _logger.LogWarning("High threat level content detected: {Preview}", 
                    input.Length > 100 ? input[..100] + "..." : input);
                
                result.IsValid = false;
                result.SanitizedContent = "[Content removed - security risk detected]";
                result.IsModified = true;
                result.SecurityIssues.Add("High-risk content patterns detected");
                return result;
            }

            // Step 2: HTML encode if no HTML is intended
            if (contentType == ContentType.PlainText)
            {
                result.SanitizedContent = HttpUtility.HtmlEncode(input);
                result.IsModified = input != result.SanitizedContent;
                result.IsValid = true;
                return result;
            }

            // Step 3: Parse and sanitize HTML
            var sanitizedHtml = SanitizeHtmlContent(input, result);
            
            // Step 4: Final validation
            result.SanitizedContent = sanitizedHtml;
            result.IsValid = true;

            // Step 5: Check for modifications
            result.IsModified = input != result.SanitizedContent;

            if (result.IsModified)
            {
                _logger.LogInformation("Content was sanitized. Original length: {Original}, Sanitized length: {Sanitized}", 
                    input.Length, result.SanitizedContent.Length);
            }

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sanitizing content");
            
            // Fallback to HTML encoding on error
            result.SanitizedContent = HttpUtility.HtmlEncode(input);
            result.IsModified = true;
            result.IsValid = true;
            result.SecurityIssues.Add("Sanitization error - fallback to HTML encoding");
            return result;
        }
    }

    public string SanitizePlainText(string input)
    {
        if (string.IsNullOrEmpty(input))
            return string.Empty;

        // Remove potential HTML/script content
        var cleaned = Regex.Replace(input, @"<[^>]*>", "", RegexOptions.IgnoreCase);
        
        // Decode HTML entities
        cleaned = HttpUtility.HtmlDecode(cleaned);
        
        // Re-encode for safe display
        return HttpUtility.HtmlEncode(cleaned).Trim();
    }

    public bool IsContentSafe(string input)
    {
        if (string.IsNullOrEmpty(input))
            return true;

        var threatLevel = DetectThreatLevel(input);
        return threatLevel != ThreatLevel.High;
    }

    public string CreateBiblicalQuote(string verse, string reference)
    {
        // Safely format biblical quotes with proper HTML structure
        var sanitizedVerse = SanitizePlainText(verse);
        var sanitizedReference = SanitizePlainText(reference);

        return $"<blockquote class=\"biblical-quote\">" +
               $"<em>\"{sanitizedVerse}\"</em>" +
               $"<footer class=\"blockquote-footer\">{sanitizedReference}</footer>" +
               $"</blockquote>";
    }

    private string SanitizeHtmlContent(string input, SanitizationResult result)
    {
        var doc = new HtmlDocument();
        doc.LoadHtml(input);

        // Remove dangerous nodes
        RemoveDangerousNodes(doc, result);

        // Sanitize remaining nodes
        SanitizeNodes(doc.DocumentNode, result);

        return doc.DocumentNode.InnerHtml;
    }

    private void RemoveDangerousNodes(HtmlDocument doc, SanitizationResult result)
    {
        var dangerousTags = new[] { "script", "iframe", "object", "embed", "form", "input", "button", "link", "meta", "style" };
        
        foreach (var tagName in dangerousTags)
        {
            var nodes = doc.DocumentNode.SelectNodes($"//{tagName}");
            if (nodes != null)
            {
                foreach (var node in nodes.ToList())
                {
                    result.SecurityIssues.Add($"Removed dangerous tag: {node.Name}");
                    node.Remove();
                }
            }
        }
    }

    private void SanitizeNodes(HtmlNode node, SanitizationResult result)
    {
        if (node.NodeType == HtmlNodeType.Element)
        {
            // Remove disallowed tags
            if (!_allowedTags.Contains(node.Name))
            {
                result.SecurityIssues.Add($"Removed disallowed tag: {node.Name}");
                node.Name = "span"; // Convert to safe span
            }

            // Sanitize attributes
            var attributesToRemove = new List<HtmlAttribute>();
            
            foreach (var attribute in node.Attributes)
            {
                if (!IsAttributeAllowed(node.Name, attribute.Name))
                {
                    attributesToRemove.Add(attribute);
                    result.SecurityIssues.Add($"Removed disallowed attribute: {attribute.Name}");
                }
                else
                {
                    // Sanitize attribute values
                    var originalValue = attribute.Value;
                    attribute.Value = SanitizeAttributeValue(attribute.Value);
                    
                    if (originalValue != attribute.Value)
                    {
                        result.SecurityIssues.Add($"Sanitized attribute value: {attribute.Name}");
                    }
                }
            }

            foreach (var attr in attributesToRemove)
            {
                node.Attributes.Remove(attr);
            }
        }

        // Recursively sanitize child nodes
        foreach (var child in node.ChildNodes.ToList())
        {
            SanitizeNodes(child, result);
        }
    }

    private bool IsAttributeAllowed(string tagName, string attributeName)
    {
        // Check tag-specific attributes first
        if (_tagSpecificAttributes.TryGetValue(tagName, out var tagAttrs))
        {
            if (tagAttrs.Contains(attributeName))
                return true;
        }

        // Check global allowed attributes
        return _allowedAttributes.Contains(attributeName);
    }

    private string SanitizeAttributeValue(string value)
    {
        if (string.IsNullOrEmpty(value))
            return value;

        // Remove dangerous patterns from attribute values
        foreach (var pattern in _maliciousPatterns)
        {
            value = pattern.Replace(value, "");
        }

        return value.Trim();
    }

    private ThreatLevel DetectThreatLevel(string input)
    {
        if (string.IsNullOrEmpty(input))
            return ThreatLevel.None;

        var issueCount = 0;
        var hasHighRiskPatterns = false;

        foreach (var pattern in _maliciousPatterns)
        {
            var matches = pattern.Matches(input);
            if (matches.Count > 0)
            {
                issueCount += matches.Count;
                
                // JavaScript and event handlers are high risk
                if (pattern.ToString().Contains("javascript") || pattern.ToString().Contains("on\\w+"))
                {
                    hasHighRiskPatterns = true;
                }
            }
        }

        if (hasHighRiskPatterns || issueCount > 3)
            return ThreatLevel.High;
        
        if (issueCount > 0)
            return ThreatLevel.Medium;

        return ThreatLevel.Low;
    }
}

public class SanitizationResult
{
    public string OriginalContent { get; set; } = string.Empty;
    public string SanitizedContent { get; set; } = string.Empty;
    public bool IsModified { get; set; }
    public bool IsValid { get; set; }
    public ThreatLevel ThreatLevel { get; set; }
    public List<string> SecurityIssues { get; set; } = new();
}

public enum ContentType
{
    General,
    PlainText,
    RichText,
    BiblicalQuote,
    UserComment
}

public enum ThreatLevel
{
    None,
    Low,
    Medium,
    High
}