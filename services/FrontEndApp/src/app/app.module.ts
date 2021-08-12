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

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    ExperimentCreationComponent,
    ExperimentMonitoringComponent,
    FilterPipe,
    ArchiveExperimentComponent,
  ],
  imports: [
    BrowserModule,
    CommonModule,
    FormsModule,
    AppRoutingModule,
    HttpClientModule,
    DragDropModule,
    ReactiveFormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
