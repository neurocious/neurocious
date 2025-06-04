using BlessedRSI.Web.Models;

namespace BlessedRSI.Web.Utilities;

public static class SubscriptionTierHelper
{
    public static string GetDisplayName(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => "Sparrow",
            SubscriptionTier.Lion => "Lion",
            SubscriptionTier.Eagle => "Eagle",
            SubscriptionTier.Shepherd => "Shepherd",
            _ => tier.ToString()
        };
    }

    public static string GetDescription(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => "Free tier - \"Consider the sparrows...\" (Luke 12:24)",
            SubscriptionTier.Lion => "$29/month - \"Bold as a lion\" (Proverbs 28:1)",
            SubscriptionTier.Eagle => "$99/month - \"Soar on wings like eagles\" (Isaiah 40:31)",
            SubscriptionTier.Shepherd => "$299/month - Ultimate leadership and guidance tier",
            _ => tier.ToString()
        };
    }

    public static string GetBiblicalReference(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => "Luke 12:24 - \"Consider the ravens: They do not sow or reap, they have no storeroom or barn; yet God feeds them. And how much more valuable you are than birds!\"",
            SubscriptionTier.Lion => "Proverbs 28:1 - \"The wicked flee though no one pursues, but the righteous are as bold as a lion.\"",
            SubscriptionTier.Eagle => "Isaiah 40:31 - \"But those who hope in the Lord will renew their strength. They will soar on wings like eagles; they will run and not grow weary, they will walk and not be faint.\"",
            SubscriptionTier.Shepherd => "John 10:11 - \"I am the good shepherd. The good shepherd lays down his life for the sheep.\"",
            _ => ""
        };
    }

    public static string GetIconClass(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => "fas fa-dove",
            SubscriptionTier.Lion => "fas fa-crown",
            SubscriptionTier.Eagle => "fas fa-feather-alt",
            SubscriptionTier.Shepherd => "fas fa-cross",
            _ => "fas fa-user"
        };
    }

    public static string GetPrice(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => "Free",
            SubscriptionTier.Lion => "$29/month",
            SubscriptionTier.Eagle => "$99/month",
            SubscriptionTier.Shepherd => "$299/month",
            _ => "Custom"
        };
    }

    public static string GetBadgeClass(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => "badge bg-secondary",
            SubscriptionTier.Lion => "badge bg-warning text-dark",
            SubscriptionTier.Eagle => "badge bg-primary",
            SubscriptionTier.Shepherd => "badge bg-success",
            _ => "badge bg-light text-dark"
        };
    }

    public static List<string> GetFeatures(SubscriptionTier tier)
    {
        return tier switch
        {
            SubscriptionTier.Sparrow => new List<string>
            {
                "Basic strategy testing",
                "Limited backtests per day",
                "Community access",
                "Educational content"
            },
            SubscriptionTier.Lion => new List<string>
            {
                "Unlimited strategy testing",
                "Advanced indicators",
                "Strategy sharing",
                "Priority support",
                "Biblical devotionals"
            },
            SubscriptionTier.Eagle => new List<string>
            {
                "Everything in Lion tier",
                "Advanced analytics",
                "Custom indicators",
                "Portfolio optimization",
                "Market alerts",
                "Live trading signals"
            },
            SubscriptionTier.Shepherd => new List<string>
            {
                "Everything in Eagle tier",
                "Full API access",
                "Custom integrations",
                "Personal mentorship",
                "Priority feature requests",
                "White-label solutions"
            },
            _ => new List<string>()
        };
    }
}