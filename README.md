# ADR-001: Batch-Oriented Brand-Preserving Video Regeneration Pipeline

## AI Writing Assistance Note

AI has been used to help with the writing of this document. The ideas and reasoning are my own, and I am happy to go into depth on any part of it at later stages. I wanted to flesh the design out a little more before I was satisfied with the submission.

## Status

Proposed

## Context

We are building an internal AI content-production engine for short-form vertical video regeneration.

The engine takes approved source clips, lets an operator select clips, regenerates the first frame using a registered brand identity, and then sends that result into a motion-video provider to produce the final 9:16 output.

The core problem is not only making the pipeline run. The harder problem is making it run at volume without losing:

- brand identity
- source motion intent
- output quality
- cost control
- auditability
- resumability

The system will process many source clips across many registered brand identities.

A batch is therefore not just a list of clips.

```text
selected source clips × selected brand identities
```

For example:

```text
200 source clips × 8 brands = 1,600 batch items
```

Each item may go through frame selection, image generation, image QC, motion generation, motion QC, storage, and review.

The main design assumption is:

```text
provider success != usable output
```

A provider can return `success` while still producing an output that is wrong for the product.

Examples:

- valid frame, but unusable as conditioning input
- good-looking image, but wrong identity
- correct first frame, but identity drifts during motion
- brand is preserved, but source action is lost
- fallback provider returns lower semantic fidelity
- output is technically valid but bad crop, blur, or low quality

Because of this, the pipeline needs explicit gates between each stage.

```text
source clip
→ source / frame QC
→ image generation
→ image identity QC
→ motion generation
→ motion / quality QC
→ operator review for exceptions
```

The operator should not need to watch every output. The system should auto-pass clear successes, block clear failures, and surface uncertain cases for review.

---

## Decision

We will design the system around a persisted batch item state machine.

The atomic unit of work is:

```text
batch_item = source_clip_id + brand_id
```

Each batch item moves independently through the pipeline.

We will persist state after every stage, store provider job IDs, store prompt and brand versions, run QC before expensive downstream stages, and make retry/fallback behaviour explicit.

The system will treat QC as part of the product pipeline, not as a manual afterthought.

---

# 1. Failure Modes Under Real Volume

## 1.1 Bad source frames will poison the whole chain

The first thing I would expect to break at real volume is not the download or ffmpeg itself.

The more common failure is that the source clip is technically valid, but the frame we pass into image generation is a bad conditioning frame.

That matters because the generated first frame becomes the anchor for the rest of the pipeline. If that anchor is wrong, the motion stage is already starting from bad input.

Example:

```text
source clip is valid
ffmpeg extracts a frame
image provider accepts the frame
motion provider accepts the generated image
final video is produced
but the output is wrong because the conditioning frame was bad
```

So frame extraction should not be treated as a utility function. It should be a real pipeline stage with its own persisted decision.

### 1.1.1 Technical method

For each source clip, the worker should extract multiple candidate frames instead of using the first frame.

Candidate timestamps:

```text
0.5s
1.0s
25% of duration
50% of duration
75% of duration
95% of duration
```

This can be done with ffmpeg:

```bash
ffmpeg -ss <timestamp> -i input.mp4 -frames:v 1 -q:v 2 candidate_<n>.jpg
```

Each candidate frame is then scored before one is selected.

Persist one row per candidate frame:

```text
candidate_frame_id
source_clip_id
timestamp_ms
r2_key
width
height
blur_score
brightness_score
black_frame_score
subject_count
main_subject_bbox
pose_visibility_score
selected_boolean
reject_reason
```

The output of this stage is:

```text
selected_frame_artifact_id
frame_qc_scores
```

If no frame is usable, the item should stop here:

```text
state = SKIPPED_INVALID_SOURCE
failure_reason = no_usable_conditioning_frame
```

The point is to avoid spending image and motion provider cost on a clip that has already failed as an input.

---

## 1.2 Frame QC should use simple CV first, then pose detection

The first pass should be cheap computer vision checks. This does not need a heavy model.

Use OpenCV-style checks for:

```text
blur_score
brightness_score
black_frame_score
```

The blur score can be computed using variance of Laplacian.

Input:

```text
candidate frame image
```

Output:

```text
single numeric sharpness score
```

Decision:

```text
if blur_score < threshold:
  reject candidate frame
```

Brightness and black-frame checks are also simple image-statistics checks.

Input:

```text
candidate frame image
```

Output:

```text
brightness_mean
black_pixel_ratio
```

Decision:

```text
if black_pixel_ratio is high:
  reject candidate frame

if brightness_mean is too low or too high:
  reject candidate frame
```

After that, run pose detection only on the candidate frames that passed the cheap checks.

For this I would use MediaPipe Pose Landmarker.

Input:

```text
candidate frame image
```

Model output:

```text
pose landmarks
x/y coordinates normalised to image width and height
z estimate
visibility score per landmark
```

Persist:

```text
pose_keypoints_json
pose_visibility_score
main_subject_bbox
```

Example stored output:

```json
{
  "timestamp_ms": 1000,
  "landmarks": [
    {
      "name": "left_shoulder",
      "x": 0.41,
      "y": 0.32,
      "z": -0.08,
      "visibility": 0.94
    },
    {
      "name": "right_shoulder",
      "x": 0.58,
      "y": 0.33,
      "z": -0.07,
      "visibility": 0.93
    }
  ]
}
```

The key thing is that pose detection is not being used because it is fancy. It is being used because we need to know whether the frame gives us a stable human structure that can later be compared against the generated video.

If the pose landmarks are weak or missing, then this frame is a poor anchor for motion preservation.

Decision:

```text
if pose_visibility_score < threshold:
  reject candidate frame or mark as weak_motion_anchor
```

Candidate frame selection should then be a scored decision:

```text
frame_score =
  blur_score
+ brightness_score
+ pose_visibility_score
+ crop_safety_score
- black_frame_penalty
- multi_subject_penalty
```

The selected frame is the highest-scoring frame that passes the hard gates.

---

## 1.3 Image generation can succeed while changing the identity

The next failure mode is semantic, not operational.

The image provider can return `success`, but the generated first frame may not preserve the registered brand identity.

That is the point where the pipeline should stop. If the image identity is already wrong, motion generation will only make the mistake more expensive.

So the image stage should look like this:

```text
IMAGE_SUBMITTED
→ IMAGE_GENERATED
→ IMAGE_IDENTITY_QC
→ IMAGE_QC_PASSED
→ MOTION_SUBMITTED
```

### 1.3.1 Face identity check

If the brand identity is a person, use InsightFace / ArcFace embeddings.

The method is:

```text
1. Detect and align the face in approved brand reference images.
2. Compute an ArcFace embedding for each approved reference.
3. Store those embeddings against the brand_identity_version.
4. Detect and align the face in the generated first frame.
5. Compute the generated frame embedding.
6. Compare generated embedding against the approved reference embeddings using cosine similarity.
```

Input:

```text
approved brand reference images
generated first frame
```

Model output:

```text
face bounding box
face embedding vector
```

Pipeline output:

```text
face_detected
face_count
face_similarity_to_brand_centroid
face_similarity_to_best_reference
```

Persist:

```text
brand_id
brand_identity_version
image_artifact_id
face_model_version
face_similarity_to_brand_centroid
face_similarity_to_best_reference
identity_threshold_used
identity_decision
```

Decision:

```text
if brand requires face and no face is detected:
  fail image QC

if multiple faces are detected and the brand expects one person:
  send to operator review

if face_similarity_to_brand_centroid < threshold:
  fail image QC
```

The threshold should be calibrated per brand, not set globally forever.

For a new brand, start with a conservative threshold and adjust once there are approved and rejected examples.

---

## 1.4 Brand/product similarity should use image embeddings, not only face matching

Face matching does not solve every brand.

Some brands are not just a face. They may depend on:

```text
logo
product shape
mascot
clothing
colour palette
packaging
overall visual identity
```

For this broader check I would use OpenCLIP image embeddings.

The method is:

```text
1. Compute image embeddings for approved brand reference images.
2. Compute image embedding for the generated first frame.
3. Compare generated embedding against approved references.
4. Also compare against known rejected examples when available.
```

Input:

```text
approved brand images
rejected brand examples if available
generated first frame
```

Model output:

```text
image embedding vector
```

Pipeline output:

```text
clip_similarity_to_approved_refs
clip_similarity_to_rejected_refs
brand_similarity_margin
```

The important score is not just:

```text
similarity to approved references
```

It is the margin:

```text
similarity_to_approved - similarity_to_rejected
```

Decision:

```text
if similarity_to_approved is low:
  send to review or fail

if generated image is closer to rejected examples than approved examples:
  fail image QC

if margin is small:
  send to review
```

This is realistic because we do not need to train a new model at the start. We can use OpenCLIP as a feature extractor and build the decision layer ourselves.

Persist:

```text
clip_model_version
approved_reference_ids_used
rejected_reference_ids_used
clip_similarity_to_approved
clip_similarity_to_rejected
brand_similarity_margin
decision
```

This gives us a cheap, repeatable way to detect when a provider has produced something that looks polished but is no longer the registered brand.

---

## 1.5 Motion generation can preserve the first frame but drift later

The motion provider can start from a good first frame and still drift over the video.

This is a different failure from image QC.

Image QC answers:

```text
Did the generated first frame preserve the brand?
```

Motion QC answers:

```text
Did the video keep that identity and preserve the source motion?
```

The dangerous case is:

```text
first frame looks correct
thumbnail looks correct
video drifts halfway through
operator does not notice unless they watch the whole clip
```

So motion QC should sample the generated video across time.

Sample frames:

```text
0%
25%
50%
75%
100%
```

Do the same for the source video where motion comparison is required.

Persist sampled frames:

```text
sample_frame_id
video_artifact_id
timestamp_ms
r2_key
sample_type = source | generated
```

---

## 1.6 Motion identity drift check

For person-based identities, run the same ArcFace identity check on each sampled generated frame.

Input:

```text
brand reference face embeddings
generated video sampled frames
```

Model output per sampled frame:

```text
face embedding
face bbox
face detection confidence
```

Pipeline output:

```text
identity_score_0
identity_score_25
identity_score_50
identity_score_75
identity_score_100
identity_min_score
identity_mean_score
identity_variance
```

The key score is:

```text
identity_min_score
```

Not the average.

Reason:

```text
A video with four good sampled frames and one broken sampled frame is still unsafe to auto-pass.
```

Decision:

```text
if identity_min_score < threshold:
  state = NEEDS_OPERATOR_REVIEW
  failure_reason = motion_identity_drift

if identity_variance is high:
  state = NEEDS_OPERATOR_REVIEW
  failure_reason = unstable_identity
```

Persist:

```text
motion_artifact_id
face_model_version
sampled_frame_ids
identity_scores_by_timestamp
identity_min_score
identity_mean_score
identity_variance
decision
```

This is how we avoid approving videos where the first frame is fine but the identity breaks during motion.

---

## 1.7 Source motion preservation should use pose trajectories

For clips where the source is meant to act as a motion template, we need to check whether the generated video preserved the action.

This should not be done by comparing pixels. The identity, clothing, background, and style are supposed to change.

Instead, compare pose trajectories.

Use MediaPipe Pose Landmarker on sampled frames from both the source and generated video.

Input:

```text
sampled source frames
sampled generated frames
```

Model output:

```text
pose landmarks per frame
x/y/z coordinates
visibility score per landmark
```

Then normalise the keypoints before comparison.

Do not compare raw pixel coordinates directly because the generated person may have different scale or position.

Normalize by body box:

```text
normalised_x = (keypoint_x - body_center_x) / body_bbox_width
normalised_y = (keypoint_y - body_center_y) / body_bbox_height
```

Then compare the same keypoints across time.

Useful keypoints:

```text
shoulders
elbows
wrists
hips
knees
ankles
nose / head position where visible
```

Pipeline output:

```text
pose_trajectory_similarity
subject_position_similarity
pose_visibility_score_source
pose_visibility_score_generated
```

Example decision:

```text
if pose_visibility_score_source is low:
  do not hard fail on pose
  mark motion_qc_confidence = low

if pose_visibility_score_generated is low:
  send to review

if pose_trajectory_similarity < threshold:
  send to review for source_motion_not_preserved
```

This is important: pose QC should know when it is uncertain.

If the source pose was not detectable, the system should not pretend it has a reliable motion comparison.

Persist:

```text
source_pose_keypoints_json
generated_pose_keypoints_json
normalisation_method
pose_trajectory_similarity
motion_qc_confidence
decision
```

This gives us a practical motion-preservation check without needing to train a full video model upfront.

---

## 1.8 Technical video quality should be checked after motion generation

Even if identity and motion pass, the final video can still be technically bad.

At this stage I would use two layers.

First, cheap deterministic checks:

```text
duration
resolution
aspect ratio
fps
black frame ratio
blur across sampled frames
brightness across sampled frames
```

Input:

```text
final generated video
```

Output:

```text
duration_seconds
width
height
fps
black_frame_ratio
blur_min_score
brightness_min_score
```

Decision:

```text
if duration is wrong:
  fail

if aspect ratio is not 9:16:
  fail

if black_frame_ratio > threshold:
  fail

if blur_min_score < threshold:
  send to review or fail
```

Second, use DOVER as a no-reference video quality score.

Input:

```text
final generated video
```

Model output:

```text
technical quality score
aesthetic quality score
overall quality score
```

Pipeline output:

```text
technical_quality_score
aesthetic_quality_score
quality_decision
```

Decision:

```text
if deterministic checks fail:
  fail or review

if deterministic checks pass but DOVER score is low:
  send to review
```

I would not make DOVER the only gate. It should be a learned quality signal on top of hard checks.

Persist:

```text
video_quality_model_version
technical_quality_score
aesthetic_quality_score
hard_quality_checks_json
quality_decision
```

---

## 1.9 Final gating logic for Section 1

The failure mode we are protecting against is not one thing. It is a chain of successful-looking steps that produce a bad output.

So the item should only move forward when the previous stage has produced a usable artifact.

The gate sequence should be:

```text
SOURCE_VALIDATED
→ FRAME_SELECTED
→ IMAGE_GENERATED
→ IMAGE_IDENTITY_QC_PASSED
→ MOTION_GENERATED
→ MOTION_QC_PASSED
```

Hard stop before image generation:

```text
no usable frame
bad crop
black/blurred conditioning frame
no detectable subject where subject is required
```

Hard stop before motion generation:

```text
generated first frame does not match brand identity
face identity below threshold
brand/product embedding too far from approved references
generated frame closer to rejected examples than approved examples
```

Review after motion generation:

```text
identity drift across sampled frames
pose trajectory does not match source
technical quality is low
motion QC confidence is low
```

The main principle:

```text
Do not let an expensive downstream stage hide an upstream failure.
```

Each gate should emit:

```text
raw model outputs
normalised scores
thresholds used
model versions
decision
failure reason
```

That gives us a pipeline that can be debugged, tuned, and compared over time instead of a black-box generation chain.

---

# 2. Batch Controls

## 2.1 Batch structure

A batch is created from:

```text
source_clip_ids[]
brand_ids[]
prompt_pack_version
provider_policy_version
fallback_policy_version
qc_policy_version
```

The system expands this into individual items:

```text
batch_item = source_clip_id + brand_id
```

Each item runs independently.

---

## 2.2 Batch item state machine

Each item should be modelled as a state machine.

Happy path:

```text
CREATED
→ SOURCE_VALIDATED
→ FRAME_SELECTED
→ IMAGE_SUBMITTED
→ IMAGE_GENERATED
→ IMAGE_QC_PASSED
→ MOTION_SUBMITTED
→ MOTION_GENERATED
→ MOTION_QC_PASSED
→ READY_FOR_REVIEW
→ APPROVED
```

Exception states:

```text
SKIPPED_INVALID_SOURCE
FAILED_RETRYABLE
FAILED_TERMINAL
NEEDS_OPERATOR_REVIEW
CANCELLED
```

This gives us:

- resumability
- auditability
- clear retry behaviour
- operator visibility
- safe recovery after worker crashes
- no need to restart full batches from scratch

---

## 2.3 State to persist

Persist the following records:

```text
batches
batch_items
stage_attempts
provider_jobs
artifacts
qc_results
operator_events
prompt_pack_snapshots
brand_identity_snapshots
```

For each `batch_item`, persist:

```text
batch_item_id
batch_id
source_clip_id
brand_id
current_state
locked_by_worker_id
state_updated_at
attempt_count
current_stage
selected_frame_artifact_id
image_artifact_id
motion_artifact_id
current_provider
provider_job_id
prompt_pack_version
brand_identity_version
qc_policy_version
failure_reason
operator_decision
```

For each `stage_attempt`, persist:

```text
stage_attempt_id
batch_item_id
stage
provider
attempt_number
idempotency_key
provider_job_id
input_artifact_id
output_artifact_id
rendered_prompt
provider_settings_json
status
error_type
error_message
started_at
completed_at
cost_estimate
```

The item stores current state.

The attempt table stores every provider attempt.

This matters because a batch item may have multiple attempts across the same provider or fallback providers.

---

## 2.4 Retry policy

Separate infrastructure failures from semantic failures.

Automatically retry infrastructure failures:

```text
network timeout
temporary provider error
upload failure
download failure
worker crash
provider status unknown
rate limit / backoff
```

Default policy:

```text
same provider retry: max 2 attempts
backoff: exponential + jitter
fallback: only after retry exhaustion or provider outage
```

Do not blindly retry semantic failures:

```text
identity mismatch
wrong logo / product
bad crop
motion drift
wrong action
source timing not preserved
low technical quality
```

These should usually go to:

```text
NEEDS_OPERATOR_REVIEW
```

Unless there is a known corrective path.

Examples:

```text
frame QC failed
→ try another candidate frame

image crop failed
→ regenerate once with crop-safe prompt modifier

image identity failed
→ try stricter identity prompt once, then review

provider outage
→ fallback provider

motion infra failed
→ retry / poll / fallback if configured

motion semantic failed
→ review, not blind retry
```

---

## 2.5 Provider fallback policy

Image provider chain:

```text
Nano Banana Pro
→ Flux Kontext Max
→ Seedream v4.5
```

Rules:

- retry the same provider first for infrastructure failures
- use fallback only after retry exhaustion or provider outage
- record fallback usage per item and per batch
- apply stricter QC to fallback outputs
- flag the batch if fallback usage is too high, for example over 20%

Motion provider chain:

```text
WaveSpeed / Kling v3 Standard
→ fallback only if configured and cost allows
```

Motion fallback should be more conservative than image fallback.

Motion generation is more expensive and more likely to introduce identity drift.

---

## 2.6 Resumability

If a worker crashes after submitting a motion job, the system should not regenerate from scratch.

Example persisted state:

```text
state = MOTION_SUBMITTED
provider_job_id = stored
```

On resume:

```text
poll provider_job_id

if complete:
  store artifact
  continue to MOTION_QC

if still running:
  stay in MOTION_SUBMITTED

if failed:
  classify as retryable / terminal / fallback
```

Provider submission should be idempotent.

Suggested idempotency key:

```text
batch_item_id + stage + attempt_number + prompt_pack_version + provider
```

The system should never rerun a completed stage unless the operator explicitly asks for it.

---

## 2.7 Operator experience during a batch

The operator should see batch progress, not raw queue noise.

Batch dashboard:

```text
total items
completed
approved
needs review
skipped invalid source
failed terminal
in progress
estimated cost so far
fallback usage %
average time per stage
```

Grid view:

```text
rows = source clips
columns = brand identities
cell = state + thumbnail + failure badge
```

Review filters:

```text
identity failure
motion drift
bad crop
low quality
provider failed
fallback used
high-cost item
```

Operator actions:

```text
approve
reject
skip
rerun same provider
rerun with fallback
choose different source frame
edit prompt modifier
send to manual review
cancel remaining batch
```

The operator should mainly handle exceptions.

They should not be the primary QC system.

---

# 3. Prompt Versioning, Brand Preservation, and Drift Detection

## 3.1 Prompt versioning

Prompts should be versioned as a full `prompt_pack`, not as loose text fields.

A prompt pack includes:

```text
image_prompt_version
image_negative_prompt_version
motion_prompt_version
motion_negative_prompt_version
provider_adapter_version
brand_identity_version
provider_settings_version
fallback_policy_version
qc_policy_version
transformation_contract_version
```

Store both:

```text
template_version
rendered_prompt
```

The rendered prompt matters because the template may be the same while the final inserted values are different.

Rollback should be pointer-based:

```text
current_prompt_pack_version = v12
rollback_to = v10
```

Do not overwrite old prompt versions.

Mark bad versions inactive.

---

## 3.2 Brand preservation evaluation

Brand preservation should be checked against a gold set.

For each brand:

```text
approved reference images
approved past generated outputs
rejected outputs
locked attributes
flexible attributes
```

The pipeline checks identity twice:

```text
generated first frame → brand identity QC
generated video frames → identity drift QC
```

Useful scores:

```text
face_similarity_to_brand
face_similarity_to_first_frame
identity_min_score
identity_mean_score
identity_variance_score
brand_asset_similarity
logo / product presence
```

Minimum score matters more than average score.

For evaluation, use a mix of:

- InsightFace / ArcFace-style embeddings where the brand identity is a person
- OpenCLIP for broader brand, product, style, and image similarity
- MediaPipe Pose Landmarker for pose and body-keypoint tracking
- VideoMAE / VideoMAE V2-style video embeddings for higher-level action similarity
- cheap CV checks first for technical quality
- DOVER-style no-reference video quality scoring where learned quality scoring is useful

This should not be one magic evaluator.

The better approach is:

```text
cheap deterministic checks
+ embedding-based checks
+ operator labels
```

---

## 3.3 Source-structure consistency

Do not compare the source video and generated video globally.

The system is supposed to change identity and style.

So comparison should be based on a transformation contract.

Example contract:

```text
preserve:
  - action
  - pose trajectory
  - timing
  - broad composition
  - camera movement where relevant

replace:
  - person identity
  - clothing
  - brand assets
  - colour palette

allow_change:
  - background
  - lighting
  - texture
```

Score only what should have been preserved:

```text
pose_trajectory_similarity
subject_position_similarity
camera_motion_similarity
action_embedding_similarity
timing_similarity
lip_sync_score_if_applicable
```

This avoids punishing the model for changes it was supposed to make.

---

## 3.4 Silent drift detection

Silent drift means the pipeline still runs, but quality slowly gets worse.

Track metrics by:

```text
brand_id
source_account_id
provider
motion_provider
prompt_pack_version
fallback_used
qc_policy_version
```

Signals to watch:

```text
identity_min_score p10 / p50 / p90
identity_variance
image QC failure rate
motion QC failure rate
operator rejection rate
manual review rate
fallback rate
retry rate
provider latency
cost per approved output
canary score movement
```

Use canary sets:

```text
same source clip
same brand
same prompt pack
same provider policy
same QC policy
```

Run these regularly.

If the inputs are fixed but scores move, something changed:

```text
provider behaviour changed
prompt adapter changed
QC model changed
brand pack changed
```

Initial version:

```text
fixed thresholds + alerting + review queue
```

Later version:

```text
QC scores + metadata + operator labels
→ logistic regression / LightGBM calibrator
→ auto-pass / review / fail
```

Operator labels should be structured:

```text
approved
rejected_identity
rejected_motion_drift
rejected_bad_crop
rejected_low_quality
rejected_wrong_action
rejected_wrong_brand_asset
```

The calibrator should not replace the QC system.

It should learn from operator decisions and help reduce review load over time.

---

# 4. Three Questions Before Starting

## 4.1 What is the exact identity contract per brand?

We need approved and rejected examples for each brand.

For each brand, define what is locked and what can change.

Example:

```text
face must stay fixed
logo must stay fixed
product shape must stay fixed
background can change
pose can change
lighting can change
```

Without this, “brand preservation” is too vague.

---

## 4.2 Is the source clip a strict motion template or a loose creative reference?

This changes the QC logic.

If the source is a strict template, then action, timing, pose, and camera movement matter a lot.

If it is only a loose reference, then the system should be more tolerant and mainly check identity and general quality.

---

## 4.3 What is the business priority order?

We need the priority order between:

```text
identity quality
speed
cost
low operator review volume
```

This controls:

```text
fallback policy
QC thresholds
auto-pass rules
how often to rerun
how much motion spend is acceptable
how conservative the review queue should be
```

This should be confirmed before locking batch policy.

---

# 5. Consequences

## 5.1 Positive consequences

- batches are resumable
- failed items do not block the whole batch
- provider success is not treated as product success
- expensive motion generation is avoided when image QC already failed
- prompt and provider regressions can be traced
- operators review exceptions instead of every output
- fallback usage and cost drift are visible
- canary sets give an early warning for silent drift

---

## 5.2 Negative consequences

- more state needs to be persisted
- more tables and state transitions need to be maintained
- QC thresholds will need calibration
- early versions may send too many items to manual review
- brand gold sets need to be created and maintained
- provider-specific prompt adapters add complexity

---

## 5.3 Trade-off

The added complexity is worth it because the alternative is a pipeline that appears to work but quietly produces wrong or unauditable outputs at volume.

For this product, silent semantic failure is worse than visible system failure.

---

# 6. Alternatives Considered

## 6.1 Alternative 1: Simple linear pipeline

```text
source clip
→ first frame
→ image generation
→ motion generation
→ operator review
```

Rejected.

This is simpler, but it wastes provider spend and pushes too much QC work onto operators.

It also does not handle resumability, fallback control, or silent drift well.

---

## 6.2 Alternative 2: Operator reviews every output

Rejected.

This may work at low volume, but it does not scale.

The operator should handle edge cases, not act as the main quality-control system.

---

## 6.3 Alternative 3: One global AI judge

Rejected.

A single judge model is too opaque and too hard to debug.

The better system is a combination of:

```text
deterministic checks
embedding checks
brand-specific gold sets
operator labels
calibrated thresholds
```

---

# 7. Final Decision Summary

Use a persisted state-machine design where each `source_clip_id + brand_id` pair is a resumable batch item.

Gate the pipeline with QC at frame, image, and motion stages.

Version prompts, provider settings, brand identity packs, fallback policies, and QC policies together as a `prompt_pack`.

Use gold sets and sampled-frame scoring to detect identity drift.

Use canary runs and quality metrics to detect silent drift over time.

Keep operators focused on exceptions, not full manual review.
