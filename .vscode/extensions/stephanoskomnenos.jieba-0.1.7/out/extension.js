"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = require("vscode");
const command_1 = require("./command");
function activate(context) {
    context.subscriptions.push(vscode.commands.registerCommand("jieba.forwardWord", command_1.forwardWord));
    context.subscriptions.push(vscode.commands.registerCommand("jieba.backwardWord", command_1.backwardWord));
    context.subscriptions.push(vscode.commands.registerCommand("jieba.killWord", command_1.killWord));
    context.subscriptions.push(vscode.commands.registerCommand("jieba.backwardKillWord", command_1.backwardKillWord));
    context.subscriptions.push(vscode.commands.registerCommand("jieba.selectWord", command_1.selectWord));
}
exports.activate = activate;
// This method is called when your extension is deactivated
function deactivate() { }
exports.deactivate = deactivate;
//# sourceMappingURL=extension.js.map