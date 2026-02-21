import { app } from "../../../scripts/app.js";

// ============================================================================
// DIT Tensor Debiaser â€” FLUX.2 Klein UI (201 Tensors)
//
// Colors by tensor type:
//   Blue   (#4488ff) = img stream tensors
//   Purple (#aa66dd) = txt stream tensors  
//   Orange (#dd8833) = single block tensors (joint cross-modal)
//   Green  (#66aa66) = global / timing / final
// ============================================================================

const N_DOUBLE = 8;
const N_SINGLE = 24;

// --- Build all 201 tensor widget names (matching Python KEY_TO_WIDGET) ---
const GLOBAL_TENSORS = [
    "img_in_weight",
    "txt_in_weight",
    "time_in_in_layer_weight",
    "time_in_out_layer_weight",
    "double_stream_modulation_img_lin_weight",
    "double_stream_modulation_txt_lin_weight",
    "single_stream_modulation_lin_weight",
    "final_layer_adaLN_modulation_1_weight",
    "final_layer_linear_weight",
];

const DB_TENSOR_SUFFIXES = [
    "img_attn_qkv_weight",
    "img_attn_proj_weight",
    "img_attn_norm_key_norm_scale",
    "img_attn_norm_query_norm_scale",
    "img_mlp_0_weight",
    "img_mlp_2_weight",
    "txt_attn_qkv_weight",
    "txt_attn_proj_weight",
    "txt_attn_norm_key_norm_scale",
    "txt_attn_norm_query_norm_scale",
    "txt_mlp_0_weight",
    "txt_mlp_2_weight",
];

const SB_TENSOR_SUFFIXES = [
    "linear1_weight",
    "linear2_weight",
    "norm_key_norm_scale",
    "norm_query_norm_scale",
];

const DB_TENSORS = [];
for (let i = 0; i < N_DOUBLE; i++) {
    for (const suffix of DB_TENSOR_SUFFIXES) {
        DB_TENSORS.push(`double_blocks_${i}_${suffix}`);
    }
}

const SB_TENSORS = [];
for (let i = 0; i < N_SINGLE; i++) {
    for (const suffix of SB_TENSOR_SUFFIXES) {
        SB_TENSORS.push(`single_blocks_${i}_${suffix}`);
    }
}

const ALL_TENSORS = [...GLOBAL_TENSORS, ...DB_TENSORS, ...SB_TENSORS];

// --- Colors ---
function tensorColor(name) {
    // Global
    if (name.startsWith("img_in")) return "#4488ff";
    if (name.startsWith("txt_in")) return "#aa66dd";
    if (name.startsWith("time_in")) return "#66aa66";
    if (name.startsWith("double_stream_modulation_img")) return "#4488ff";
    if (name.startsWith("double_stream_modulation_txt")) return "#aa66dd";
    if (name.startsWith("single_stream_modulation")) return "#dd8833";
    if (name.startsWith("final_layer")) return "#66aa66";
    
    // Double blocks
    if (name.includes("img_attn")) return "#4488ff";
    if (name.includes("img_mlp")) return "#3377dd";
    if (name.includes("txt_attn")) return "#aa66dd";
    if (name.includes("txt_mlp")) return "#9955cc";
    
    // Single blocks
    if (name.startsWith("single_blocks")) return "#dd8833";
    
    return "#888";
}

function tensorBg(name, on) {
    if (!on) return "#1a1a1a";
    if (name.includes("img_") && !name.startsWith("single")) return "#1c2030";
    if (name.includes("txt_") && !name.startsWith("single")) return "#241c2e";
    if (name.startsWith("single_blocks") || name.startsWith("single_stream")) return "#2a2214";
    return "#252525";
}

// --- Extract block number for grouping ---
function getBlockGroup(name) {
    if (GLOBAL_TENSORS.includes(name)) return "global";
    
    let m = name.match(/^double_blocks_(\d+)_/);
    if (m) return `db${m[1]}`;
    
    m = name.match(/^single_blocks_(\d+)_/);
    if (m) return `sb${m[1]}`;
    
    return "other";
}

// ============================================================================
// EXTENSION
// ============================================================================

app.registerExtension({
    name: "FluxKleinTensorDebiaser.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxKleinTensorDebiaser") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                if (node._tensorInit) return;
                node._tensorInit = true;
                node._setupCombinedWidgets();
                if (node.size[0] < 580) node.size[0] = 580;
                node.setDirtyCanvas(true);
            }, 50);
        };

        const origConf = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origConf) origConf.apply(this, arguments);
            setTimeout(() => this.setDirtyCanvas(true), 120);
        };

        const origExec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (out) {
            if (origExec) origExec.apply(this, arguments);
            if (out?.analysis_json?.[0]) {
                try { this._ad = JSON.parse(out.analysis_json[0]); } catch {}
                this.setDirtyCanvas(true);
            }
        };

        // ----- Combined checkbox + slider for each tensor -----
        nodeType.prototype._setupCombinedWidgets = function () {
            const strNames = new Set(
                this.widgets.filter(w => w.name.endsWith("_str")).map(w => w.name)
            );

            for (const w of this.widgets) {
                const sn = w.name + "_str";
                if (!strNames.has(sn)) continue;
                const sw = this.widgets.find(ww => ww.name === sn);
                if (!sw) continue;
                this._makeCombined(w, sw, w.name);
            }

            this.setSize(this.computeSize());
        };

        nodeType.prototype._makeCombined = function (toggle, strength, name) {
            const M = 6, CB = 12, GAP = 4, LW = 200, VW = 40;
            const MIN = -2.0, MAX = 2.0, STEP = 0.01;

            toggle.draw = function (ctx, node, ww, y, wh) {
                const SW = ww - M - CB - GAP - LW - GAP - VW - M - GAP;
                const cbX = M;
                const labelX = cbX + CB + GAP;
                const sliderX = labelX + LW + GAP;

                const on = Boolean(toggle.value);
                let str = parseFloat(strength.value);
                if (isNaN(str)) str = 1.0;

                const col = tensorColor(name);

                // Row bg
                ctx.fillStyle = tensorBg(name, on);
                ctx.fillRect(0, y, ww, wh);

                // Checkbox
                ctx.strokeStyle = on ? col : "#555";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(cbX, y + (wh - CB) / 2, CB, CB);
                if (on) {
                    ctx.fillStyle = col;
                    ctx.fillRect(cbX + 2, y + (wh - CB) / 2 + 2, CB - 4, CB - 4);
                }

                // Label (truncate if needed)
                ctx.globalAlpha = on ? 1.0 : 0.35;
                ctx.fillStyle = on ? "#ddd" : "#666";
                ctx.font = "10px monospace";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                
                // Shorten label for display
                let lbl = name;
                // Remove common prefixes for brevity
                lbl = lbl.replace("double_blocks_", "db");
                lbl = lbl.replace("single_blocks_", "sb");
                lbl = lbl.replace("_weight", "");
                lbl = lbl.replace("_scale", "");
                lbl = lbl.replace("double_stream_modulation_", "dsm_");
                lbl = lbl.replace("single_stream_modulation_", "ssm_");
                lbl = lbl.replace("final_layer_", "final_");
                lbl = lbl.replace("adaLN_modulation_1", "adaLN");
                
                while (ctx.measureText(lbl + "â€¦").width > LW - 4 && lbl.length > 8)
                    lbl = lbl.slice(0, -1);
                if (lbl.length < name.length - 20) lbl += "â€¦";
                
                ctx.fillText(lbl, labelX, y + wh / 2);
                ctx.globalAlpha = 1.0;

                // Slider track
                const ty = y + wh / 2, th = 4;
                const range = MAX - MIN;
                const norm = (str - MIN) / range;
                const zNorm = (0 - MIN) / range;

                ctx.fillStyle = "#333";
                ctx.beginPath();
                ctx.roundRect(sliderX, ty - th / 2, SW, th, 2);
                ctx.fill();

                if (on) {
                    const zX = sliderX + zNorm * SW;
                    const sX = sliderX + norm * SW;
                    ctx.fillStyle = str >= 0 ? col : "#ff6655";
                    ctx.beginPath();
                    ctx.roundRect(Math.min(zX, sX), ty - th / 2, Math.abs(sX - zX), th, 2);
                    ctx.fill();
                }

                // Thumb
                ctx.fillStyle = on ? "#fff" : "#555";
                ctx.beginPath();
                ctx.arc(sliderX + norm * SW, ty, 4, 0, Math.PI * 2);
                ctx.fill();

                // Value
                ctx.fillStyle = on ? "#ccc" : "#555";
                ctx.textAlign = "right";
                ctx.font = "9px monospace";
                ctx.fillText(str.toFixed(2), ww - M, y + wh / 2);
            };

            const origMouse = toggle.mouse?.bind(toggle);
            toggle.mouse = function (event, pos, node) {
                const ww = node.size[0];
                const SW = ww - M - CB - GAP - LW - GAP - VW - M - GAP;
                const sliderX = M + CB + GAP + LW + GAP;
                const lx = pos[0];

                if (lx >= sliderX - 4 && lx <= sliderX + SW + 4) {
                    if (event.type === "pointerdown" || event.type === "pointermove") {
                        let n = Math.max(0, Math.min(1, (lx - sliderX) / SW));
                        let v = MIN + n * (MAX - MIN);
                        v = Math.round(v / STEP) * STEP;
                        v = Math.max(MIN, Math.min(MAX, v));
                        strength.value = v;
                        node.setDirtyCanvas(true);
                        return true;
                    }
                }

                if (origMouse) return origMouse(event, pos, node);
                return false;
            };

            // Hide the strength widget row
            strength.draw = function () {};
            strength.computeSize = function () { return [0, -4]; };
        };
    }
});
