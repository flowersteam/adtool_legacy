import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { HttpClientModule } from '@angular/common/http';

import { AppComponent } from './app.component';
import { HomeComponent } from './home/home.component';
import { ExperimentCreationComponent } from './experiment-creation/experiment-creation.component';
import { ExperimentMonitoringComponent } from './experiment-monitoring/experiment-monitoring.component';
import { ArchiveExperimentComponent } from './experiment-monitoring/archive-experiment/archive-experiment.component';

import {FilterPipe} from './pipes/filter.pipe';
import { DragDropModule }from '@angular/cdk/drag-drop';
import { ReactiveFormsModule } from '@angular/forms';

import { NgxSliderModule } from '@angular-slider/ngx-slider';
import { ToastrModule } from 'ngx-toastr';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { SelectInputAccordingToTypeComponent } from './experiment-creation/select-input-according-to-type/select-input-according-to-type.component';
import { SetDiscoverySavingKeyComponent } from './experiment-creation/set-discovery-saving-key/set-discovery-saving-key.component';
import { SetModuleComponent } from './experiment-creation/set-module/set-module.component';
import { SetModuleListComponent } from './experiment-creation/set-module-list/set-module-list.component';
import { DisplayInputspaceOutputspaceComponent } from './experiment-creation/display-inputspace-outputspace/display-inputspace-outputspace.component';
import { SetExperimentConfigComponent } from './experiment-creation/set-experiment-config/set-experiment-config.component';
import { LoadExperimentConfigToCreateComponent } from './experiment-creation/load-experiment-config-to-create/load-experiment-config-to-create.component';
import { SetModuleConfigComponent } from './experiment-creation/set-module-config/set-module-config.component';
import { LogsComponent } from './experiment-monitoring/logs/logs.component';
import { JupyterLabComponent } from './experiment-monitoring/jupyter-lab/jupyter-lab.component';
import { DiscoveryComponent } from './experiment-monitoring/discovery/discovery.component';
import { CheckpointsDetailsComponent } from './experiment-monitoring/checkpoints-details/checkpoints-details.component';
import { ExperimentConfigDetailsComponent } from './experiment-monitoring/experiment-config-details/experiment-config-details.component';
import { MultiCollapserComponent } from './experiment-monitoring/multi-collapser/multi-collapser.component';
import { CheckboxListComponent } from './experiment-monitoring/checkbox-list/checkbox-list.component';
import { FilterComponent } from './experiment-monitoring/filter/filter.component';
import { MonacoEditorModule } from 'ngx-monaco-editor';
import { CodeEditorComponent } from './experiment-creation/code-editor/code-editor.component';
import { SetDictionaryComponent } from './experiment-creation/set-dictionary/set-dictionary.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    ExperimentCreationComponent,
    ExperimentMonitoringComponent,
    FilterPipe,
    ArchiveExperimentComponent,
    SelectInputAccordingToTypeComponent,
    SetDiscoverySavingKeyComponent,
    SetModuleComponent,
    SetModuleListComponent,
    DisplayInputspaceOutputspaceComponent,
    SetExperimentConfigComponent,
    LoadExperimentConfigToCreateComponent,
    SetModuleConfigComponent,
    LogsComponent,
    JupyterLabComponent,
    DiscoveryComponent,
    CheckpointsDetailsComponent,
    ExperimentConfigDetailsComponent,
    MultiCollapserComponent,
    CheckboxListComponent,
    FilterComponent,
    CodeEditorComponent,
    SetDictionaryComponent,
  ],
  imports: [
    BrowserModule,
    CommonModule,
    FormsModule,
    AppRoutingModule,
    HttpClientModule,
    DragDropModule,
    ReactiveFormsModule,
    NgxSliderModule,
    ToastrModule.forRoot(),
    BrowserAnimationsModule,
    MonacoEditorModule.forRoot(),
  ],
  providers: [SetModuleComponent],
  bootstrap: [AppComponent]
})
export class AppModule { }
