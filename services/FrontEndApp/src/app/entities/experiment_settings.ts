import { ExplorerSettings } from '../entities/explorer_settings';
import { SystemSettings } from '../entities/system_settings';
import { InputWrapperSettings } from '../entities/input_wrapper_settings';
import { OutputRepresentationSettings } from '../entities/output_representation_settings';
import { Callback } from '../entities/callback';
import { Config } from '../entities/config';

export interface ExperimentSettings {
    experiment: Config;
    explorer: Config;
    system: Config;
    input_wrappers: Config[];
    output_representations: Config[];
    callbacks: string[];
    logger_handlers: any[];
    host:string;
  }