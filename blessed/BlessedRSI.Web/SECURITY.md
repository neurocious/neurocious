# BlessedRSI Security Implementation

## 🔐 **Production-Grade JWT Authentication**

### **Architecture Overview**
- **JWT Access Tokens**: Short-lived (15 minutes) for API authentication
- **HTTP-Only Refresh Tokens**: Secure, long-lived (30 days) stored in cookies
- **Role-Based Authorization**: Fine-grained permissions with subscription tiers
- **Security Event Logging**: Comprehensive audit trail for all authentication events

### **Key Security Features**

#### **1. Token Management**
```csharp
// Access Token Configuration
- Expiration: 15 minutes (production) / 60 minutes (development)
- Algorithm: HMAC-SHA256
- Claims: User ID, email, roles, subscription tier, community points
- Issuer/Audience validation enabled
- Clock skew: Zero tolerance
```

#### **2. Password Security**
```csharp
// ASP.NET Identity Configuration
- Minimum 8 characters
- Requires: uppercase, lowercase, digit, special character
- Account lockout: 5 failed attempts = 15 minute lockout
- Unique email requirement
- Password history prevention
```

#### **3. Refresh Token Security**
```csharp
// Refresh Token Features
- Cryptographically secure random generation (64 bytes)
- Database storage with expiration tracking
- Automatic cleanup of expired tokens
- Single-use tokens (revoked after refresh)
- IP address tracking for suspicious activity
```

#### **4. Security Headers** (Production)
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Content-Security-Policy: [Comprehensive CSP policy]
```

#### **5. CORS Configuration**
```csharp
// Production CORS Policy
- Specific origin allowlist
- Credentials enabled for cookie auth
- Method and header restrictions
- No wildcard origins in production
```

### **Authentication Flow**

#### **Registration Process**
1. **Input Validation**: Server-side validation with regex patterns
2. **Email Uniqueness**: Database-level constraints
3. **Password Hashing**: ASP.NET Identity with PBKDF2
4. **Role Assignment**: Default "User" role assignment
5. **Token Generation**: Immediate JWT creation for seamless login
6. **Security Logging**: Registration event with IP/UserAgent tracking

#### **Login Process**
1. **Credential Validation**: Secure password comparison
2. **Account Lockout**: Automatic protection against brute force
3. **Token Generation**: Access + Refresh token pair
4. **Cookie Security**: HTTP-only, Secure, SameSite=Strict
5. **Failed Attempt Tracking**: Incremental lockout with reset on success

#### **Token Refresh Process**
1. **Expired Token Validation**: Claims extraction without lifetime validation
2. **Refresh Token Verification**: Database lookup with active status check
3. **Token Rotation**: Old refresh token revoked, new pair generated
4. **Security Audit**: All refresh attempts logged with client details

### **Authorization Policies**

#### **Role-Based Access**
```csharp
[Authorize(Policy = "RequireUser")]          // Basic authenticated users
[Authorize(Policy = "RequireAdmin")]         // Administrative functions
[Authorize(Policy = "RequireBelieversOrHigher")] // Paid subscription features
```

#### **Subscription-Based Access**
```csharp
// Subscription Tier Enforcement
- Seeker (Free): 3 backtests/day, community access
- Believer ($29): Unlimited backtests, advanced analytics
- Disciple ($99): Real-time E-RSI, custom alerts  
- Apostle ($299): API access, white-label features
```

### **Security Monitoring**

#### **Security Event Types**
```csharp
public enum SecurityEventType
{
    Login,                    // Successful/failed login attempts
    Logout,                   // User-initiated logouts
    PasswordChange,           // Password modification events
    PasswordReset,            // Password reset requests/completions
    AccountLockout,           // Automatic account locks
    SuspiciousActivity,       // Invalid tokens, unusual patterns
    TokenRefresh,             // Token refresh operations
    Registration,             // New account creation
    EmailConfirmation,        // Email verification events
    TwoFactorEnabled,         // 2FA activation (future)
    TwoFactorDisabled         // 2FA deactivation (future)
}
```

#### **Audit Trail Features**
- **IP Address Tracking**: Client IP for all security events
- **User Agent Logging**: Browser/device identification
- **Geolocation Ready**: Prepared for location-based security
- **Retention Policy**: 60-day security event retention
- **Automated Cleanup**: Expired token cleanup with audit preservation

### **Data Protection**

#### **Database Security**
```csharp
// Entity Framework Security
- Connection string encryption
- Parameterized queries (SQL injection prevention)
- Database-level constraints
- Cascade delete policies for data integrity
```

#### **Sensitive Data Handling**
```csharp
// Protected Information
- Refresh tokens: Cryptographically secure random generation
- User passwords: ASP.NET Identity hashing (PBKDF2)
- API keys: Configuration-based with environment separation
- Personal data: GDPR-compliant data handling patterns
```

### **Rate Limiting & DoS Protection**

#### **API Rate Limiting**
```csharp
// Request Throttling (Basic Implementation)
- API endpoint monitoring with IP-based tracking
- Logarithmic backoff for failed authentication
- Configurable limits per subscription tier
- Future: Distributed rate limiting with Redis
```

#### **Subscription-Based Limits**
```csharp
// Tier-Based Resource Limits
- Seeker: 3 backtests/day, basic community access
- Believer+: Unlimited backtests, priority processing
- API rate limits scale with subscription tier
```

### **Development vs Production**

#### **Development Settings**
```json
{
  "JwtSettings": {
    "AccessTokenExpirationMinutes": 60,  // Extended for development
    "RefreshTokenExpirationDays": 7,     // Shorter for testing
    "SecretKey": "Development-key-32-chars-minimum"
  }
}
```

#### **Production Settings**
```json
{
  "JwtSettings": {
    "AccessTokenExpirationMinutes": 15,  // Security-focused short expiry
    "RefreshTokenExpirationDays": 30,    // Balanced user experience
    "SecretKey": "Production-Crypto-Strong-Key-64-Characters-Plus"
  }
}
```

### **Security Best Practices Implemented**

#### **✅ OWASP Top 10 Compliance**
1. **Injection Prevention**: Parameterized queries, input validation
2. **Broken Authentication**: JWT best practices, secure session management
3. **Sensitive Data Exposure**: Encryption at rest and in transit
4. **XML External Entities**: Not applicable (JSON-based API)
5. **Broken Access Control**: Role-based authorization with claims
6. **Security Misconfiguration**: Hardened default configurations
7. **XSS Protection**: Content Security Policy, input sanitization
8. **Insecure Deserialization**: Safe JSON handling, type validation
9. **Known Vulnerabilities**: Regular dependency updates
10. **Insufficient Logging**: Comprehensive security event logging

#### **✅ Additional Security Measures**
- **Defense in Depth**: Multiple security layers
- **Principle of Least Privilege**: Minimal permission grants
- **Secure by Default**: Safe configuration defaults
- **Fail Securely**: Graceful error handling without information disclosure
- **Security Testing**: Input validation, boundary testing
- **Incident Response**: Security event monitoring and alerting ready

### **Future Security Enhancements**

#### **Planned Features**
1. **Two-Factor Authentication**: TOTP/SMS integration
2. **Device Fingerprinting**: Advanced session security
3. **Geolocation Monitoring**: Location-based anomaly detection
4. **Advanced Rate Limiting**: Distributed Redis-based throttling
5. **Webhook Security**: Signed payload verification for Stripe
6. **API Key Management**: Rotating keys for external integrations
7. **Penetration Testing**: Regular security assessments

#### **Compliance Readiness**
- **GDPR**: Data portability, right to erasure, consent management
- **SOC 2**: Security controls documentation
- **PCI DSS**: Payment card security (via Stripe compliance)
- **CCPA**: California privacy law compliance

### **Security Contact**
For security issues or vulnerability reports:
- **Email**: security@blessedrsi.com
- **Response Time**: 24 hours for critical issues
- **Disclosure Policy**: Responsible disclosure encouraged

---

*"The simple believe anything, but the prudent give thought to their steps." - Proverbs 14:15*

**Built with security, guided by wisdom.**