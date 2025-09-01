<script lang="ts">
    import OpenRecentFile from "./OpenRecentFile";
    import {App, TFile} from "obsidian";
    import {onMount} from "svelte";
    import {MarkdownRenderer} from "obsidian";

    export let plugin: Plugin;
    export let files: string[] = [];
    export let app: App

    // async function readFile(path: string) {
    //     const file = app.vault.getAbstractFileByPath(path)
    //     if (file instanceof TFile) {
    //         lastFileContent = (await app.vault.read(file))
    //     }
    // }

    let previewContainer: HTMLDivElement;

    onMount(async () => {
        const file = app.vault.getAbstractFileByPath(files[0])
        if (file instanceof TFile) {
            const content = await app.vault.read(file)

            await MarkdownRenderer.renderMarkdown(
                content,
                previewContainer,
                file,
                plugin
            )
        }
    })
</script>

<section class="container">
    <h1>Recent Files</h1>
    <h5>Last 10 used files</h5>
    <hr>
    <section class="list">
        {#each files as file}
            {#if file === files[0]}
                <section class="listElement">
                    <section class="viewArea">
                        <button id="fileName" onclick={() => OpenRecentFile(file)}>{file}</button>
                    </section>
                    <div class="previewArea" bind:this={previewContainer}></div>
                </section>
            {:else}
                <section class="viewArea">
                    <button id="fileName" onclick={() => OpenRecentFile(file)}>{file}</button>
                </section>
            {/if}
        {/each}
    </section>
</section>

<!--<div class="box"> <div class="group">-->
<!--        <div class>Fundamentals.md</div>-->
<!--        <button class="button" onclick={() => {OpenRecentFile(files[0])}}>preview</button>-->
<!--    </div>-->
<!--</div>-->

<style lang="postcss">
    @reference "tailwindcss";

    h1 {
        margin: 10px;
    }

    h5 {
        margin: 0 0 0 10px;
        font-size: small;
        font-weight: normal;
    }

    #fileName {
        all: unset;
        cursor: pointer;
    }

    hr {
        margin: 10px;
    }

    .container {
        overflow: scroll;
        border: 1px gray solid;
        padding: 10px;
        display: flex;
        flex-direction: column;
        width: 50%;
    }

    .list {
        margin-left: 10px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }

    .listElement {
        margin: 0;
        font-size: small;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .previewArea {
        overflow-y: scroll;
        width: 100%;
        text-align: left;
        height: 200px;
        font-size: 0.85rem;
        line-height: 1.3;
        border-bottom: 1px solid black;
    }


    .viewArea {
        padding: 10px;
        font-size: small;
        border-bottom: 1px solid black;
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
