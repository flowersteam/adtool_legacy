import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError, map } from 'rxjs/operators';

import { ExperimentSettings } from '../../entities/experiment_settings';
import { ExplorerSettings } from '../../entities/explorer_settings';
import { SystemSettings } from '../../entities/system_settings';
import { InputWrapperSettings } from '../../entities/input_wrapper_settings';
import { OutputRepresentationSettings } from '../../entities/output_representation_settings';
import { Callback } from '../../entities/callback';
import { RESTResponse, httpErrorResponseToRESTResponse, httpResponseToRESTResponse } from '../../entities/rest_response';

import { Observable, of } from 'rxjs';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AutoDiscServerService {

  private autodiscServerUrl

  constructor(private http: HttpClient) {
    this.autodiscServerUrl = "http://" + environment.GATEWAY_HOST + ":" + environment.GATEWAY_PORT + "/autodisc-server";
    console.log("autodiscServerUrl:" + this.autodiscServerUrl);
  }

  getExplorers(): Observable<RESTResponse<ExplorerSettings[]>> {
    return this.http.get<ExplorerSettings[]>(
      this.autodiscServerUrl + "/explorers", { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<ExplorerSettings[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<ExplorerSettings[]>(response)); })
      );
  }

  getInputWrappers(): Observable<RESTResponse<InputWrapperSettings[]>> {
    return this.http.get<InputWrapperSettings[]>(
      this.autodiscServerUrl + "/input-wrappers", { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<InputWrapperSettings[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<InputWrapperSettings[]>(response)); })
      );
  }

  getSystems(): Observable<RESTResponse<SystemSettings[]>> {
    return this.http.get<SystemSettings[]>(
      this.autodiscServerUrl + "/systems", { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<SystemSettings[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<SystemSettings[]>(response)); })
      );
  }

  getOutputRepresentations(): Observable<RESTResponse<OutputRepresentationSettings[]>> {
    return this.http.get<OutputRepresentationSettings[]>(
      this.autodiscServerUrl + "/output-representations", { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<OutputRepresentationSettings[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<OutputRepresentationSettings[]>(response)); })
      );
  }

  getDiscoverySavingKeys(explorer: string): Observable<RESTResponse<string[]>> {
    return this.http.get<string[]>(
      this.autodiscServerUrl + "/discovery-saving-keys" + "/" + explorer, { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<string[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<string[]>(response)); })
      );
  }

  getCallbacks(): Observable<RESTResponse<Callback[]>> {
    return this.http.get<Callback[]>(
      this.autodiscServerUrl + "/callbacks", { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<Callback[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<Callback[]>(response)); })
      );
  }

  getHosts(): Observable<RESTResponse<string[]>> {
    return this.http.get<string[]>(
      this.autodiscServerUrl + "/hosts", { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<string[]>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<string[]>(response)); })
      );
  }

  createExperiment(newExperiment: ExperimentSettings): Observable<RESTResponse<any>> {
    return this.http.post<ExperimentSettings>(this.autodiscServerUrl + "/experiments", newExperiment, { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<any>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<ExplorerSettings[]>(response)); })
      );
  }

  stopExperiment(id: number): Observable<RESTResponse<any>> {
    return this.http.delete<any>(this.autodiscServerUrl + "/experiments/" + id, { observe: 'response' })
      .pipe(
        map(response => { return httpResponseToRESTResponse<any>(response); }),
        catchError(response => { return of(httpErrorResponseToRESTResponse<ExplorerSettings[]>(response)); })
      );
  }
}
