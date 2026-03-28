# Changelog

## March 2026

### SDK

- **Default backend URL**: updated to `https://api.axioniclabs.ai`
- **Password reset flow**: `change_password()` triggers server-side reset email instead of direct password change; `change-email` CLI command deprecated in favor of Spectra dashboard
- **Runtime policy API**: create, update, delete, and list policies in both local and remote modes
- **Bug fixes**: SAE hooks with TransformerLens, `whoami` auth header, steering vector layer key resolution, policy CRUD in local mode
