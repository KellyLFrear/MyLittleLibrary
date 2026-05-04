/**
 * Login / Register page logic.
 *
 * Handles form submission for both Login and Register modes,
 * calling the backend API and redirecting to main_page.html on success.
 */

const form           = document.getElementById('login-form');
const titleEl        = document.getElementById('auth-title');
const submitBtn      = document.getElementById('auth-submit-btn');
const toggleLink     = document.getElementById('auth-toggle-link');
const toggleText     = document.getElementById('auth-toggle-text');
const rememberRow    = document.getElementById('remember-row');
const errorEl        = document.getElementById('auth-error');
const usernameInput  = document.getElementById('auth-username');
const passwordInput  = document.getElementById('auth-password');

let isRegisterMode = false;

// ── Toggle between Login and Register ────────────────────────────────────────

toggleLink.addEventListener('click', (e) => {
    e.preventDefault();
    isRegisterMode = !isRegisterMode;

    if (isRegisterMode) {
        titleEl.textContent    = 'Register';
        submitBtn.textContent  = 'Create Account';
        toggleLink.textContent = 'Login';
        toggleText.firstChild.textContent = 'Already have an account? ';
        rememberRow.style.display = 'none';
    } else {
        titleEl.textContent    = 'Login';
        submitBtn.textContent  = 'Login';
        toggleLink.textContent = 'Register';
        toggleText.firstChild.textContent = "Don't have an account? ";
        rememberRow.style.display = '';
    }

    errorEl.style.display = 'none';
    errorEl.textContent   = '';
});

// ── Form submission ───────────────────────────────────────────────────────────

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();

    errorEl.style.display = 'none';
    errorEl.textContent   = '';
    submitBtn.disabled    = true;

    const endpoint = isRegisterMode ? '/api/auth/register' : '/api/auth/login';

    try {
        const res  = await fetch(endpoint, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ username, password }),
        });
        const data = await res.json();

        if (!res.ok) {
            errorEl.textContent   = data.error || 'Something went wrong.';
            errorEl.style.display = 'block';
        } else {
            // Success — navigate to the main application page
            window.location.href = '/main_page.html';
        }
    } catch (err) {
        errorEl.textContent   = 'Could not reach the server. Is the app running?';
        errorEl.style.display = 'block';
    } finally {
        submitBtn.disabled = false;
    }
});
