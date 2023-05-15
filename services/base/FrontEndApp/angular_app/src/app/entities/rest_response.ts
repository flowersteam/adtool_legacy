import { HttpErrorResponse, HttpResponse } from '@angular/common/http';

export interface RESTResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
}

export function httpResponseToRESTResponse<T>(
  response: HttpResponse<T>
): RESTResponse<T> {
  return {
    success: response.ok,
    data: response.body,
    message: undefined,
  } as RESTResponse<T>;
}

export function httpErrorResponseToRESTResponse<T>(
  response: HttpErrorResponse
): RESTResponse<T> {
  let message: string =
    typeof response.error === 'string' ? response.error : response.message;
  if (response.status === 0) {
    // A client-side or network error occurred. Handle it accordingly.
    message = 'An error occurred:' + message;
  } else {
    // The backend returned an unsuccessful response code.
    // The response body may contain clues as to what went wrong.
    message = `Backend returned code ${response.status}, body was: ` + message;
  }

  console.log(message);

  return {
    success: response.ok,
    data: undefined,
    message: message,
  } as RESTResponse<T>;
}
