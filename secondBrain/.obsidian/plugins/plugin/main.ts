import {Plugin, WorkspaceLeaf} from "obsidian";
import {ExampleView, RECENT_FILES_VIEW} from "./view";

export default class Homepage extends Plugin {

	async activateView() {
		const {workspace} = this.app

		let leaf: WorkspaceLeaf | null = null

		leaf = workspace.getLeaf(true)

		await leaf.setViewState({
			type: RECENT_FILES_VIEW,
			active: true,
		})

		await workspace.revealLeaf(leaf)

	}

	async onload() {

		this.registerView(RECENT_FILES_VIEW, (leaf) => new ExampleView(leaf)
		);

		this.addRibbonIcon('dice', 'Activate View', () => {
			this.activateView()

		})

	}

}
