import {ItemView, TFile, WorkspaceLeaf} from 'obsidian';
import RecentFilesList from "./RecentFilesList.svelte";
import {mount, unmount} from 'svelte';
import {getLocalTimeZone} from "@internationalized/date";

export const RECENT_FILES_VIEW = 'Recent Files';

export class ExampleView extends ItemView {
    RecentFilesList: ReturnType<typeof RecentFilesList> | undefined;

	constructor(leaf: WorkspaceLeaf) {
		super(leaf);
	}

	getViewType() {
		return RECENT_FILES_VIEW;
	}

	getDisplayText() {
		return 'Homepage';
	}

	async onOpen() {
		const container = this.containerEl;
		container.empty();
        // getLocalTimeZone()

		this.RecentFilesList = mount(RecentFilesList, {
			target: container,
			props: {
				files: this.app.workspace.getLastOpenFiles(),
                app: this.app,
                plugin: this,
			}
		});

	}


	async onClose() {
		if (this.RecentFilesList) {
			await unmount(this.RecentFilesList);
		}
	}
}
