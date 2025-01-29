"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseSentence = void 0;
const jieba_wasm_1 = require("jieba-wasm");
;
function parseSentence(sentence) {
    return (0, jieba_wasm_1.tokenize)(sentence, "default", true);
}
exports.parseSentence = parseSentence;
//# sourceMappingURL=parse.js.map