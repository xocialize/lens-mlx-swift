// Chat-template rendering for the Lens encoder (T5) — appended by P4.
//
// The GPT-OSS harmony template applied to Lens's fixed conversation is PROMPT-AFFINE:
// rendered = PREFIX + prompt + SUFFIX (verified against transformers
// apply_chat_template with a sentinel diff; oracle in goldens/chat_template_ref.json,
// asserted by ChatTemplateTests). add_special_tokens is a no-op for this tokenizer,
// so Swift tokenizes the rendered string plainly.

extension LensChatTemplate {
    public static let renderedPrefix = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-06-12\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background.\n\n<|end|><|start|>user<|message|>"
    public static let renderedSuffix = "<|end|><|start|>assistant<|channel|>analysis<|message|>Need to generate one image according to the description.<|end|><|start|>assistant<|channel|>final<|message|>"

    /// PREFIX + prompt + SUFFIX — the exact string the encoder tokenizes.
    public static func render(prompt: String) -> String {
        renderedPrefix + prompt + renderedSuffix
    }
}
