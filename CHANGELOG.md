# Changelog

## March 27, 2026

### Bug Fixes

- Fixed 17 undefined variable errors in `steering_opt.py`: missing `import gc`, missing `vector_clamp` parameter in `melbo_loss_func`, missing `matrix_left`/`matrix_right` initialization in `optimize_vector_minibatch_hf`
- Switched CI workflows to self-hosted runner, corrected trigger branches

---

## March 2026

### SDK

- **Default backend URL**: updated to `https://api.axioniclabs.ai`
- **Runtime policy API**: create, update, delete, and list policies in both local and remote modes
- **Bug fixes**: SAE hooks with TransformerLens, `whoami` auth header, steering vector layer key resolution, policy CRUD in local mode
