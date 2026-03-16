#ifndef WORKFLOW_STATUS_H
#define WORKFLOW_STATUS_H

/**
 * @brief 工作流层统一状态码。
 *
 * 设计目的：
 * - 训练、推理、I/O 相关模块通过同一组错误码对外返回，降低调用方分支复杂度。
 */
typedef enum WorkflowStatus {
    WORKFLOW_STATUS_OK = 0,
    WORKFLOW_STATUS_INVALID_ARGUMENT = -1,
    WORKFLOW_STATUS_IO_ERROR = -2,
    WORKFLOW_STATUS_DATA_ERROR = -3,
    WORKFLOW_STATUS_INTERNAL_ERROR = -4
} WorkflowStatus;

#endif
