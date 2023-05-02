export interface Checkpoint {
  id: number;
  experiment_id: number;
  parent_id: number;
  status: number;
  error_message: string;
}
