import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { catchError, map } from 'rxjs/operators';

import { environment } from './../../../environments/environment';
import { Experiment } from '../../entities/experiment';
import {
  RESTResponse,
  httpErrorResponseToRESTResponse,
  httpResponseToRESTResponse,
} from '../../entities/rest_response';

import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AppDbService {
  private appDBUrl;
  constructor(private http: HttpClient) {
    this.appDBUrl =
      'http://' +
      environment.GATEWAY_HOST +
      ':' +
      environment.GATEWAY_PORT +
      '/app-db-api';
    console.log('appDBUrl:' + this.appDBUrl);
  }

  /** GET LightExperiments from the AppDB */
  getLightExperiments(): Observable<RESTResponse<Experiment[]>> {
    return this.http
      .get<Experiment[]>(
        this.appDBUrl +
          '/experiments?select=id,name,created_on,progress,exp_status,archived,config,' +
          'systems(name),' +
          'explorers(name),' +
          'input_wrappers(name),' +
          'output_representations(name)&archived=is.false',
        {
          headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
          observe: 'response',
        }
      )
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<Experiment[]>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<Experiment[]>(response));
        })
      );
  }

  getExperimentById(id: number): Observable<RESTResponse<Experiment>> {
    return this.http
      .get<Experiment>(
        this.appDBUrl +
          '/experiments?select=id,name,created_on,progress,exp_status,config,archived,checkpoint_saves_archived,discoveries_archived,' +
          'systems(name,config),' +
          'explorers(name,config),' +
          'input_wrappers(name,config,index),' +
          'output_representations(name,config,index),' +
          'checkpoints!experiment_id(id,parent_id,status)' +
          '&id=eq.' +
          id,
        {
          headers: new HttpHeaders({
            'Content-Type': 'application/json',
            Accept: 'application/vnd.pgrst.object+json', // Get a single json element instead of an array
          }),
          observe: 'response',
        }
      )
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<Experiment>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<Experiment>(response));
        })
      );
  }

  private patchExperimentById(
    id: number,
    patchObject: Object
  ): Observable<RESTResponse<Experiment>> {
    return this.http
      .patch<Experiment>(
        this.appDBUrl + '/experiments?id=eq.' + id,
        patchObject,
        { observe: 'response' }
      )
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<Experiment>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<Experiment>(response));
        })
      );
  }

  updateArchiveExperimentStatusById(
    id: number,
    status: boolean
  ): Observable<RESTResponse<Experiment>> {
    return this.patchExperimentById(id, { archived: status });
  }

  archiveExperimentCheckpointSavesById(
    id: number
  ): Observable<RESTResponse<Experiment>> {
    return this.patchExperimentById(id, { checkpoint_saves_archived: true });
  }

  archiveExperimentDiscoveriesById(
    id: number
  ): Observable<RESTResponse<Experiment>> {
    return this.patchExperimentById(id, { discoveries_archived: true });
  }

  getAllLogLevels(): Observable<RESTResponse<any>> {
    return this.http
      .get<any>(this.appDBUrl + '/log_levels', {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
        observe: 'response',
      })
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<any>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<any>(response));
        })
      );
  }

  getLogs(filter: string): Observable<RESTResponse<any>> {
    return this.http
      .get<any>(this.appDBUrl + '/logs' + filter, {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
        observe: 'response',
      })
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<any>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<any>(response));
        })
      );
  }

  getPreparingLogs(filter: string): Observable<any> {
    return this.http
      .get<any>(this.appDBUrl + '/preparing_logs' + filter, {
        headers: new HttpHeaders({ 'Content-Type': 'application/json' }),
        observe: 'response',
      })
      .pipe(
        map((response) => {
          return httpResponseToRESTResponse<any>(response);
        }),
        catchError((response) => {
          return of(httpErrorResponseToRESTResponse<any>(response));
        })
      );
  }
}
