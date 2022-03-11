import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';

import { ExperimentSettings } from '../entities/experiment_settings';
import { ExplorerSettings } from '../entities/explorer_settings';
import { SystemSettings } from '../entities/system_settings';
import { InputWrapperSettings } from '../entities/input_wrapper_settings';
import { OutputRepresentationSettings } from '../entities/output_representation_settings';
import { Callback } from '../entities/callback';

import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AutoDiscServerService {

  private autodiscServerUrl = "http://127.0.0.1:5000"


  httpOptions = {
    headers: new HttpHeaders({

      'Content-Type': 'application/json' })
  };

  constructor(private http: HttpClient) { }

  getExplorers(): Observable<ExplorerSettings[]> {
    return this.http.get<ExplorerSettings[]>(
      this.autodiscServerUrl + "/explorers")
      .pipe(
        catchError(this.handleError<ExplorerSettings[]>('getExplorers', []))
      );
  }

  getInputWrappers(): Observable<InputWrapperSettings[]> {
    return this.http.get<InputWrapperSettings[]>(
      this.autodiscServerUrl + "/input-wrappers")
      .pipe(
        catchError(this.handleError<InputWrapperSettings[]>('getInputWrappers', []))
      );
  }

  getSystems(): Observable<SystemSettings[]> {
    return this.http.get<SystemSettings[]>(
      this.autodiscServerUrl + "/systems")
      .pipe(
        catchError(this.handleError<SystemSettings[]>('getSystems', []))
      );
  }

  getOutputRepresentations(): Observable<OutputRepresentationSettings[]> {
    return this.http.get<OutputRepresentationSettings[]>(
      this.autodiscServerUrl + "/output-representations")
      .pipe(
        catchError(this.handleError<OutputRepresentationSettings[]>('getOutputRepresentations', []))
      );
  }

  getDiscoverySavingKeys(): Observable<String[]> {
    return this.http.get<String[]>(
      this.autodiscServerUrl + "/discovery-saving-keys")
      .pipe(
        catchError(this.handleError<String[]>('getDiscoverySavingKeys', []))
      );
  }

  getCallbacks(): Observable<Callback[]> {
    return this.http.get<Callback[]>(
      this.autodiscServerUrl + "/callbacks")
      .pipe(
        catchError(this.handleError<Callback[]>('getCallbacks', []))
      );
  }

  getHosts(): Observable<string[]> {
    return this.http.get<string[]>(
      this.autodiscServerUrl + "/hosts")
      .pipe(
        catchError(this.handleError<string[]>('getHosts', []))
      );
  }

  createExperiment(newExperiment: ExperimentSettings): Observable<any>{
    return this.http.post<ExperimentSettings>(this.autodiscServerUrl + "/experiments", newExperiment).pipe(
      catchError(this.handleError<ExperimentSettings>('createExperiment'))
      );
  }

  stopExperiment(id: number): Observable<any>{
    return this.http.delete<string>(this.autodiscServerUrl + "/experiments/" + id).pipe(
      catchError(this.handleError<string>('stopExperiment'))
      );
  }

  /**
   * Handle Http operation that failed.
   * Let the app continue.
   * @param operation - name of the operation that failed
   * @param result - optional value to return as the observable result
  */
  private handleError<T>(operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {

      // TODO: send the error to remote logging infrastructure
      console.error(error); // log to console instead

      // TODO: better job of transforming error for user consumption
      console.log(`${operation} failed: ${error.message}`);

      return of(error as T);
    };
  }
}
