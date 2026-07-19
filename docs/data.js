const BENCH = {
 "labels": {
  "legacy_mediapipe": "Mismatched indices",
  "legacy_dlib": "Opaque fill",
  "legacy_dlib_swap": "Channel swap",
  "legacy_gan": "Untrained GAN",
  "new_classic": "This engine"
 },
 "summary": {
  "legacy_mediapipe": {
   "containment": 0.3372261107116742,
   "background_integrity": 1.0,
   "lip_texture_corr": 0.8106021267739872,
   "identity_ssim": 0.9758265106246315,
   "runtime_ms": 7.417613361030817,
   "n_images": 25
  },
  "legacy_dlib": {
   "containment": 0.7914153005394347,
   "background_integrity": 0.9999809122026871,
   "lip_texture_corr": 0.8648776475676072,
   "identity_ssim": 0.9968295098562705,
   "runtime_ms": 12.775661759951618,
   "n_images": 25
  },
  "legacy_dlib_swap": {
   "containment": 0.13974413348219472,
   "background_integrity": 0.016661640720618043,
   "lip_texture_corr": 0.9081771897674257,
   "identity_ssim": 0.9925255627682119,
   "runtime_ms": 12.416773360455409,
   "n_images": 25
  },
  "legacy_gan": {
   "containment": 0.0889977541999753,
   "background_integrity": 3.427625683688909e-07,
   "lip_texture_corr": -0.20528287662158123,
   "identity_ssim": 0.5862941965236271,
   "runtime_ms": 41.159700120042544,
   "n_images": 25
  },
  "new_classic": {
   "containment": 0.8249081709134104,
   "background_integrity": 1.0,
   "lip_texture_corr": 0.9956192955837928,
   "identity_ssim": 0.9984990197558185,
   "runtime_ms": 285.1643967203563,
   "n_images": 25
  }
 },
 "original_protocol": {
  "legacy_mediapipe": {
   "ssim": 0.5203447225182739,
   "accuracy": 0.68,
   "precision": 1.0
  },
  "legacy_dlib": {
   "ssim": 0.5300118690686845,
   "accuracy": 0.88,
   "precision": 1.0
  },
  "legacy_dlib_swap": {
   "ssim": 0.5272386933371618,
   "accuracy": 0.8,
   "precision": 1.0
  },
  "legacy_gan": {
   "ssim": 0.3777676968124315,
   "accuracy": 0.24,
   "precision": 1.0
  },
  "new_classic": {
   "ssim": 0.530701332583938,
   "accuracy": 0.88,
   "precision": 1.0
  }
 }
};
