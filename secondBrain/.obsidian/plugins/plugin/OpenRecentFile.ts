import {TFile, App} from "obsidian";

declare global {
	interface Window {
		app: App
	}
}

const app: App = window.app;

export default async function openRecentFile(path: string) {
	const file = app.vault.getAbstractFileByPath(path)
	if (file instanceof TFile) {
		const leaf = app.workspace.getLeaf(false)
		await leaf.openFile(file)
	} else {
		console.log('file not found')
	}
}

