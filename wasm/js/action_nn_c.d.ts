/**
 * @file action_nn_c.d.ts
 * @brief TypeScript definitions for action_nn_c WebAssembly module
 */

declare module 'action_nn_c' {
    /**
     * WebAssembly Module interface
     */
    export interface ActionNnCModule {
        // Version functions
        action_c_wasm_get_version_string(): string;
        action_c_wasm_get_version_major(): number;
        action_c_wasm_get_version_minor(): number;
        action_c_wasm_get_version_patch(): number;

        // Memory management
        action_c_wasm_malloc(size: number): number;
        action_c_wasm_free(ptr: number): void;
        action_c_wasm_get_memory_size(): number;
        action_c_wasm_get_heap_size(): number;

        // Inference API
        action_c_wasm_infer_create(configBytes: number, configSize: number): number;
        action_c_wasm_infer_run(
            ctxPtr: number,
            inputPtr: number,
            inputSize: number,
            outputPtr: number,
            outputSize: number
        ): number;
        action_c_wasm_infer_destroy(ctxPtr: number): void;

        // Training API (if enabled)
        action_c_wasm_train_create?(configBytes: number, configSize: number): number;
        action_c_wasm_train_step?(
            ctxPtr: number,
            inputPtr: number,
            inputSize: number,
            targetPtr: number,
            targetSize: number,
            learningRate: number
        ): number;
        action_c_wasm_train_get_loss?(ctxPtr: number): number;
        action_c_wasm_train_destroy?(ctxPtr: number): void;

        // Network type queries
        action_c_wasm_get_network_type_count(): number;
        action_c_wasm_get_network_type_name(index: number): string | null;
        action_c_wasm_is_network_type_enabled(name: string): number;

        // Profiler (placeholder)
        action_c_wasm_profiler_create?(): number;
        action_c_wasm_profiler_destroy?(ptr: number): void;

        // Debug logging (if enabled)
        action_c_wasm_debug_log?(message: number): void;

        // Emscripten helpers
        cwrap(ident: string, returnType: string | null, argTypes: string[]): (...args: any[]) => any;
        ccall(ident: string, returnType: string | null, argTypes: string[], args: any[]): any;
        _malloc(size: number): number;
        _free(ptr: number): void;
        HEAP8: Int8Array;
        HEAP16: Int16Array;
        HEAP32: Int32Array;
        HEAPU8: Uint8Array;
        HEAPU16: Uint16Array;
        HEAPU32: Uint32Array;
        HEAPF32: Float32Array;
        HEAPF64: Float64Array;
    }

    /**
     * Factory function to create the Wasm module
     */
    export default function ActionNnC(config?: Partial<ModuleConfig>): Promise<ActionNnCModule>;

    /**
     * Module configuration options
     */
    export interface ModuleConfig {
        /** Path to the .wasm file */
        wasmBinary?: ArrayBuffer | Uint8Array;
        
        /** Locate the .wasm file */
        locateFile?(path: string, scriptDirectory: string): string;
        
        /** Called when module is initialized */
        onRuntimeInitialized?(): void;
        
        /** Initial memory size in bytes */
        INITIAL_MEMORY?: number;
        
        /** Maximum memory size in bytes */
        MAXIMUM_MEMORY?: number;
        
        /** Allow memory to grow */
        ALLOW_MEMORY_GROWTH?: boolean;
    }

    /**
     * Helper class for managing inference
     */
    export class InferenceContext {
        constructor(module: ActionNnCModule, config: Uint8Array);
        run(input: Float32Array): Float32Array;
        destroy(): void;
    }

    /**
     * Helper class for managing training (if enabled)
     */
    export class TrainingContext {
        constructor(module: ActionNnCModule, config: Uint8Array);
        step(input: Float32Array, target: Float32Array, learningRate: number): number;
        getLoss(): number;
        destroy(): void;
    }

    /**
     * Get list of supported network types
     */
    export function getSupportedNetworks(module: ActionNnCModule): string[];

    /**
     * Check if a network type is enabled
     */
    export function isNetworkEnabled(module: ActionNnCModule, networkType: string): boolean;
}

// Global declaration for browser usage
declare global {
    interface Window {
        ActionNnC?: typeof import('action_nn_c').default;
    }
}

export {};
