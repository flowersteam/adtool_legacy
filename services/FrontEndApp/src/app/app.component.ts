import { Component } from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'Automated Discovery Tool';

  public path: string
  
  constructor(private router: Router) {
    this.path = ''
    router.events
    .subscribe((event) => {
      if (event instanceof NavigationEnd) {
        this.path = event.url
      }
    });
}

}
