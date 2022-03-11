import { Injectable } from '@angular/core';
import { ToastrService } from 'ngx-toastr';
import { NavigationEnd, NavigationStart, Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class ToasterService {

  constructor(private toastr: ToastrService, private router: Router,) {
    router.events.subscribe(
      (event) => {
          if ( event instanceof NavigationStart ) {
              this.clearAllExceptEternal()
          }
      });
   }

  options = {
    timeOut: 10000,
    extendedTimeOut: 10000,
    isCleanOnThePageChange: true,
  };

  showSuccess(message: string, title: string, options?:any){
    options = Object.assign({}, this.options, options);
    this.toastr.success(message, title, options)
  }
  
  showError(message: string, title: string, options?:any){
    options = Object.assign({}, this.options, options);
    this.toastr.error(message, title, options)
  }
  
  showInfo(message: string, title: string, options?:any){
    options = Object.assign({}, this.options, options);
    this.toastr.info(message, title, options)
  }
  
  showWarning(message: string, title: string, options?:any){
    options = Object.assign({}, this.options, options);
    this.toastr.warning(message, title, options)
  }

  clearAllExceptEternal(){
    for(let toast of this.toastr.toasts){
      if(toast.portal.instance.options.isCleanOnThePageChange > 0){
        this.toastr.clear(toast.toastId);
      }
    }
  }
}
