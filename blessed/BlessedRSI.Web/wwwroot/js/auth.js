// BlessedRSI Authentication JavaScript Helper
window.blessedRsiAuth = {
    // Store JWT token
    setToken: function (token) {
        if (token) {
            localStorage.setItem('accessToken', token);
            this.setAuthHeader(token);
        }
    },

    // Get JWT token
    getToken: function () {
        return localStorage.getItem('accessToken');
    },

    // Remove JWT token
    removeToken: function () {
        localStorage.removeItem('accessToken');
        this.removeAuthHeader();
    },

    // Set authorization header for fetch requests
    setAuthHeader: function (token) {
        if (window.fetch) {
            const originalFetch = window.fetch;
            window.fetch = function (url, options = {}) {
                if (!options.headers) {
                    options.headers = {};
                }
                
                // Add authorization header if token exists and not already present
                if (token && !options.headers['Authorization']) {
                    options.headers['Authorization'] = `Bearer ${token}`;
                }
                
                return originalFetch(url, options);
            };
        }
    },

    // Remove authorization header
    removeAuthHeader: function () {
        // Reset fetch to original implementation
        if (window.originalFetch) {
            window.fetch = window.originalFetch;
        }
    },

    // Check if token is expired
    isTokenExpired: function (token) {
        if (!token) return true;
        
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            const now = Date.now() / 1000;
            return payload.exp < now;
        } catch (e) {
            return true;
        }
    },

    // Get user info from token
    getUserFromToken: function (token) {
        if (!token) return null;
        
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            return {
                id: payload.sub,
                email: payload.email,
                name: payload.name,
                firstName: payload.first_name,
                lastName: payload.last_name,
                subscriptionTier: payload.subscription_tier,
                roles: payload.role || []
            };
        } catch (e) {
            return null;
        }
    },

    // Make authenticated API call
    fetchWithAuth: async function (url, options = {}) {
        const token = this.getToken();
        
        if (!options.headers) {
            options.headers = {};
        }
        
        if (token && !this.isTokenExpired(token)) {
            options.headers['Authorization'] = `Bearer ${token}`;
        }
        
        const response = await fetch(url, options);
        
        // If unauthorized, try to refresh token
        if (response.status === 401) {
            const refreshed = await this.refreshToken();
            if (refreshed) {
                // Retry with new token
                options.headers['Authorization'] = `Bearer ${this.getToken()}`;
                return fetch(url, options);
            } else {
                // Redirect to login
                this.redirectToLogin();
            }
        }
        
        return response;
    },

    // Refresh JWT token
    refreshToken: async function () {
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                credentials: 'include', // Include HTTP-only cookie
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    accessToken: this.getToken(),
                    refreshToken: '' // Will be read from cookie
                })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success && data.accessToken) {
                    this.setToken(data.accessToken);
                    return true;
                }
            }
        } catch (error) {
            console.error('Token refresh failed:', error);
        }
        
        return false;
    },

    // Login user
    login: async function (email, password, rememberMe = false) {
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email: email,
                    password: password,
                    rememberMe: rememberMe
                })
            });

            const data = await response.json();
            
            if (data.success && data.accessToken) {
                this.setToken(data.accessToken);
                return { success: true, user: data.user };
            } else {
                return { success: false, message: data.message, errors: data.errors };
            }
        } catch (error) {
            return { success: false, message: 'Login failed. Please try again.' };
        }
    },

    // Register user
    register: async function (userData) {
        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userData)
            });

            const data = await response.json();
            
            if (data.success && data.accessToken) {
                this.setToken(data.accessToken);
                return { success: true, user: data.user };
            } else {
                return { success: false, message: data.message, errors: data.errors };
            }
        } catch (error) {
            return { success: false, message: 'Registration failed. Please try again.' };
        }
    },

    // Logout user
    logout: async function () {
        try {
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.getToken()}`
                }
            });
        } catch (error) {
            console.error('Logout API call failed:', error);
        } finally {
            this.removeToken();
            this.redirectToLogin();
        }
    },

    // Redirect to login page
    redirectToLogin: function () {
        const currentPath = window.location.pathname;
        if (currentPath !== '/login' && currentPath !== '/register') {
            window.location.href = `/login?returnUrl=${encodeURIComponent(currentPath)}`;
        }
    },

    // Check authentication status
    isAuthenticated: function () {
        const token = this.getToken();
        return token && !this.isTokenExpired(token);
    },

    // Initialize auth system
    init: function () {
        const token = this.getToken();
        if (token) {
            if (this.isTokenExpired(token)) {
                this.removeToken();
            } else {
                this.setAuthHeader(token);
            }
        }

        // Set up automatic token refresh
        this.setupTokenRefresh();
    },

    // Setup automatic token refresh
    setupTokenRefresh: function () {
        const token = this.getToken();
        if (token && !this.isTokenExpired(token)) {
            try {
                const payload = JSON.parse(atob(token.split('.')[1]));
                const expirationTime = payload.exp * 1000;
                const now = Date.now();
                const timeUntilExpiry = expirationTime - now;
                
                // Refresh token 2 minutes before expiry
                const refreshTime = timeUntilExpiry - (2 * 60 * 1000);
                
                if (refreshTime > 0) {
                    setTimeout(() => {
                        this.refreshToken();
                    }, refreshTime);
                }
            } catch (e) {
                console.error('Error setting up token refresh:', e);
            }
        }
    }
};

// File download helper function
window.downloadFile = function (filename, content, contentType = 'text/plain') {
    const blob = new Blob([content], { type: contentType });
    const url = window.URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
};

// Initialize authentication system when page loads
document.addEventListener('DOMContentLoaded', function () {
    window.blessedRsiAuth.init();
});

// Handle page visibility change to refresh tokens when page becomes visible
document.addEventListener('visibilitychange', function () {
    if (!document.hidden && window.blessedRsiAuth.isAuthenticated()) {
        const token = window.blessedRsiAuth.getToken();
        if (window.blessedRsiAuth.isTokenExpired(token)) {
            window.blessedRsiAuth.refreshToken();
        }
    }
});